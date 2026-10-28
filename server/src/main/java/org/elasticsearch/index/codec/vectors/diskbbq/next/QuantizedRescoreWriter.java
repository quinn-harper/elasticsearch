/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexOutput;
import org.elasticsearch.index.codec.vectors.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringFloatVectorValues;
import org.elasticsearch.index.codec.vectors.cluster.ClusteringVectorValues;
import org.elasticsearch.index.codec.vectors.diskbbq.CentroidSupplier;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Writes the per vector residual codes that let the rescoring phase avoid the raw vectors. Every vector is first
 * quantized with OSQ against the centroid of the cluster it is primarily assigned to, which is the same first level
 * quantization the posting lists use, and what that leaves behind (the residual) is encoded with a
 * {@link PooledLloydMaxQuantizer}. The OSQ code has to be stored again here because the codes in the posting lists are
 * grouped by cluster, repeated for overspilled vectors, and cannot be found by vector ordinal.
 * <p>
 * Layout: {@code rescoreBits, osqBits, dimension, numberOfVectors, numberOfCentroids, clusterIdBytes} as ints, the
 * codebook and the per dimension scales as float bits, then one fixed length record per ordinal. A record holds the lower
 * and upper OSQ interval and the squared norm of the reconstructed vector as float bits, the cluster id in
 * {@code clusterIdBytes} little endian bytes, the packed OSQ code and the packed residual code. The squared norm lets
 * Euclidean and cosine scores be computed from codebook sums without decoding the vector.
 */
final class QuantizedRescoreWriter {

    private static final int MAX_TRAINING_VALUES = 4_000_000;
    private static final int MIN_TRAINING_VECTORS = 256;

    /** Where the section was written, in the output it was written to. */
    record Section(long offset, long length) {}

    private QuantizedRescoreWriter() {}

    /**
     * @param centroidSupplier the centroids the posting lists were written with
     * @param assignments the primary centroid ordinal of every vector ordinal
     * @return the written section, or {@code null} if this field cannot have one (no float vectors, no vectors, or no float centroids)
     */
    static Section write(
        FieldInfo fieldInfo,
        ClusteringVectorValues<?> vectors,
        CentroidSupplier centroidSupplier,
        int[] assignments,
        int osqBits,
        int rescoreBits,
        IndexOutput output
    ) throws IOException {
        if (rescoreBits <= 0
            || vectors.size() == 0
            || centroidSupplier.size() == 0
            || centroidSupplier.byteCentroid(0) != null
            || fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32
            || vectors instanceof ClusteringFloatVectorValues == false) {
            return null;
        }
        final ClusteringFloatVectorValues floatVectors = (ClusteringFloatVectorValues) vectors;
        final int dimension = floatVectors.dimension();
        final int numberOfVectors = floatVectors.size();
        final int numberOfCentroids = centroidSupplier.size();
        assert assignments.length == numberOfVectors : "expected [" + numberOfVectors + "] assignments but got " + assignments.length;
        final ResidualComputer residualComputer = new ResidualComputer(fieldInfo, dimension, osqBits);

        final PooledLloydMaxQuantizer quantizer = PooledLloydMaxQuantizer.fit(
            sampleResiduals(floatVectors, centroidSupplier, assignments, residualComputer, numberOfVectors, dimension),
            rescoreBits
        );
        final float[] codebook = quantizer.codebook();
        final float[] dimensionScales = quantizer.dimensionScales();

        final int clusterIdBytes = QuantizedRescoreValues.clusterIdBytes(numberOfCentroids);
        final long sectionOffset = output.getFilePointer();
        output.writeInt(rescoreBits);
        output.writeInt(osqBits);
        output.writeInt(dimension);
        output.writeInt(numberOfVectors);
        output.writeInt(numberOfCentroids);
        output.writeInt(clusterIdBytes);
        for (float codebookEntry : codebook) {
            output.writeInt(Float.floatToIntBits(codebookEntry));
        }
        for (float scale : dimensionScales) {
            output.writeInt(Float.floatToIntBits(scale));
        }

        final int osqPackedLength = BitPacking.packedLength(dimension, osqBits);
        final int clusterIdOffset = QuantizedRescoreValues.HEADER_FLOATS * Float.BYTES;
        final int osqCodesOffset = clusterIdOffset + clusterIdBytes;
        final int residualCodesOffset = osqCodesOffset + osqPackedLength;
        final byte[] recordBytes = new byte[QuantizedRescoreValues.recordLength(dimension, osqBits, rescoreBits, clusterIdBytes)];
        final ByteBuffer recordBuffer = ByteBuffer.wrap(recordBytes).order(ByteOrder.LITTLE_ENDIAN);
        final int[] residualCodes = new int[dimension];
        for (int ord = 0; ord < numberOfVectors; ord++) {
            final int cluster = assignments[ord];
            final float[] centroid = centroidSupplier.centroid(cluster);
            residualComputer.compute(floatVectors.vectorValue(ord), centroid);
            quantizer.encode(residualComputer.residual, residualCodes);

            double squaredNorm = 0;
            for (int i = 0; i < dimension; i++) {
                final float component = QuantizedRescoreValues.reconstructComponent(
                    centroid[i],
                    residualComputer.lowerInterval,
                    residualComputer.step,
                    residualComputer.osqCodes[i],
                    codebook[residualCodes[i]],
                    dimensionScales[i]
                );
                squaredNorm += (double) component * component;
            }

            recordBuffer.putFloat(0, residualComputer.lowerInterval);
            recordBuffer.putFloat(Float.BYTES, residualComputer.upperInterval);
            recordBuffer.putFloat(2 * Float.BYTES, (float) squaredNorm);
            for (int i = 0; i < clusterIdBytes; i++) {
                recordBytes[clusterIdOffset + i] = (byte) (cluster >>> (Byte.SIZE * i));
            }
            BitPacking.pack(residualComputer.osqCodes, dimension, osqBits, recordBytes, osqCodesOffset);
            BitPacking.pack(residualCodes, dimension, rescoreBits, recordBytes, residualCodesOffset);
            output.writeBytes(recordBytes, recordBytes.length);
        }
        return new Section(sectionOffset, output.getFilePointer() - sectionOffset);
    }

    private static float[][] sampleResiduals(
        ClusteringFloatVectorValues vectors,
        CentroidSupplier centroidSupplier,
        int[] assignments,
        ResidualComputer residualComputer,
        int numberOfVectors,
        int dimension
    ) throws IOException {
        final int sampleSize = Math.min(numberOfVectors, Math.max(MIN_TRAINING_VECTORS, MAX_TRAINING_VALUES / dimension));
        final int stride = Math.max(1, numberOfVectors / sampleSize);
        final float[][] sample = new float[sampleSize][];
        int collected = 0;
        for (int ord = 0; ord < numberOfVectors && collected < sampleSize; ord += stride) {
            residualComputer.compute(vectors.vectorValue(ord), centroidSupplier.centroid(assignments[ord]));
            sample[collected++] = residualComputer.residual.clone();
        }
        return collected == sampleSize ? sample : Arrays.copyOf(sample, collected);
    }

    /**
     * Quantizes a vector with OSQ against a centroid and exposes what is needed to store it: the interval, the codes and
     * the residual left after reconstructing the vector from them. The computation is deterministic, so running it again
     * on the same vector for the encoding pass gives the same residual as the training pass.
     */
    private static final class ResidualComputer {
        private final OptimizedScalarQuantizer osq;
        private final byte osqBits;
        private final int levels;
        private final float[] centered;
        final int[] osqCodes;
        final float[] residual;
        float lowerInterval;
        float upperInterval;
        float step;

        ResidualComputer(FieldInfo fieldInfo, int dimension, int osqBits) {
            this.osq = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
            this.osqBits = (byte) osqBits;
            this.levels = 1 << osqBits;
            this.centered = new float[dimension];
            this.osqCodes = new int[dimension];
            this.residual = new float[dimension];
        }

        void compute(float[] vector, float[] centroid) {
            final OptimizedScalarQuantizer.QuantizationResult result = osq.scalarQuantize(vector, centered, osqCodes, osqBits, centroid);
            lowerInterval = result.lowerInterval();
            upperInterval = result.upperInterval();
            step = (upperInterval - lowerInterval) / (levels - 1);
            for (int i = 0; i < residual.length; i++) {
                residual[i] = centered[i] - (lowerInterval + step * osqCodes[i]);
            }
        }
    }
}

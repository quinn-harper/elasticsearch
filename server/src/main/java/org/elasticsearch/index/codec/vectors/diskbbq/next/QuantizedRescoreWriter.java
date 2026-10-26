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

import java.io.IOException;
import java.util.Arrays;

/**
 * Writes the per vector residual codes that let the rescoring phase avoid the raw vectors. Every vector is first
 * quantized with OSQ against the field's global centroid, and what that leaves behind (the residual) is encoded with a
 * {@link PooledLloydMaxQuantizer}. The OSQ code has to be stored again here because the codes in the posting lists are
 * grouped by cluster, repeated for overspilled vectors, and cannot be found by vector ordinal.
 * <p>
 * Layout: {@code rescoreBits, osqBits, dimension, numberOfVectors} as ints, the codebook and the per dimension scales as
 * float bits, then one fixed length record per ordinal: lower and upper interval as float bits, the packed OSQ code and
 * the packed residual code.
 */
final class QuantizedRescoreWriter {

    private static final int MAX_TRAINING_VALUES = 4_000_000;
    private static final int MIN_TRAINING_VECTORS = 256;

    /** Where the section was written, in the output it was written to. */
    record Section(long offset, long length) {}

    private QuantizedRescoreWriter() {}

    /**
     * @return the written section, or {@code null} if this field cannot have one (no float vectors, no vectors, or no global centroid)
     */
    static Section write(
        FieldInfo fieldInfo,
        ClusteringVectorValues<?> vectors,
        float[] globalCentroid,
        int osqBits,
        int rescoreBits,
        IndexOutput output
    ) throws IOException {
        if (rescoreBits <= 0
            || globalCentroid == null
            || vectors.size() == 0
            || fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32
            || vectors instanceof ClusteringFloatVectorValues == false) {
            return null;
        }
        final ClusteringFloatVectorValues floatVectors = (ClusteringFloatVectorValues) vectors;
        final int dimension = floatVectors.dimension();
        final int numberOfVectors = floatVectors.size();
        final ResidualComputer residualComputer = new ResidualComputer(fieldInfo, globalCentroid, osqBits);

        final PooledLloydMaxQuantizer quantizer = PooledLloydMaxQuantizer.fit(
            sampleResiduals(floatVectors, residualComputer, numberOfVectors, dimension),
            rescoreBits
        );

        final long sectionOffset = output.getFilePointer();
        output.writeInt(rescoreBits);
        output.writeInt(osqBits);
        output.writeInt(dimension);
        output.writeInt(numberOfVectors);
        for (float codebookEntry : quantizer.codebook()) {
            output.writeInt(Float.floatToIntBits(codebookEntry));
        }
        for (float scale : quantizer.dimensionScales()) {
            output.writeInt(Float.floatToIntBits(scale));
        }

        final int osqPackedLength = BitPacking.packedLength(dimension, osqBits);
        final byte[] codeBytes = new byte[osqPackedLength + BitPacking.packedLength(dimension, rescoreBits)];
        final int[] residualCodes = new int[dimension];
        for (int ord = 0; ord < numberOfVectors; ord++) {
            residualComputer.compute(floatVectors.vectorValue(ord));
            quantizer.encode(residualComputer.residual, residualCodes);
            BitPacking.pack(residualComputer.osqCodes, dimension, osqBits, codeBytes, 0);
            BitPacking.pack(residualCodes, dimension, rescoreBits, codeBytes, osqPackedLength);
            output.writeInt(Float.floatToIntBits(residualComputer.lowerInterval));
            output.writeInt(Float.floatToIntBits(residualComputer.upperInterval));
            output.writeBytes(codeBytes, codeBytes.length);
        }
        return new Section(sectionOffset, output.getFilePointer() - sectionOffset);
    }

    private static float[][] sampleResiduals(
        ClusteringFloatVectorValues vectors,
        ResidualComputer residualComputer,
        int numberOfVectors,
        int dimension
    ) throws IOException {
        final int sampleSize = Math.min(numberOfVectors, Math.max(MIN_TRAINING_VECTORS, MAX_TRAINING_VALUES / dimension));
        final int stride = Math.max(1, numberOfVectors / sampleSize);
        final float[][] sample = new float[sampleSize][];
        int collected = 0;
        for (int ord = 0; ord < numberOfVectors && collected < sampleSize; ord += stride) {
            residualComputer.compute(vectors.vectorValue(ord));
            sample[collected++] = residualComputer.residual.clone();
        }
        return collected == sampleSize ? sample : Arrays.copyOf(sample, collected);
    }

    /**
     * Quantizes a vector with OSQ against the global centroid and exposes what is needed to store it: the interval, the
     * codes and the residual left after reconstructing the vector from them. The computation is deterministic, so running
     * it again on the same vector for the encoding pass gives the same residual as the training pass.
     */
    private static final class ResidualComputer {
        private final OptimizedScalarQuantizer osq;
        private final float[] globalCentroid;
        private final byte osqBits;
        private final int levels;
        private final float[] centered;
        final int[] osqCodes;
        final float[] residual;
        float lowerInterval;
        float upperInterval;

        ResidualComputer(FieldInfo fieldInfo, float[] globalCentroid, int osqBits) {
            this.osq = new OptimizedScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
            this.globalCentroid = globalCentroid;
            this.osqBits = (byte) osqBits;
            this.levels = 1 << osqBits;
            this.centered = new float[globalCentroid.length];
            this.osqCodes = new int[globalCentroid.length];
            this.residual = new float[globalCentroid.length];
        }

        void compute(float[] vector) {
            final OptimizedScalarQuantizer.QuantizationResult result = osq.scalarQuantize(vector, centered, osqCodes, osqBits, globalCentroid);
            lowerInterval = result.lowerInterval();
            upperInterval = result.upperInterval();
            final float step = (upperInterval - lowerInterval) / (levels - 1);
            for (int i = 0; i < residual.length; i++) {
                residual[i] = centered[i] - (lowerInterval + step * osqCodes[i]);
            }
        }
    }
}

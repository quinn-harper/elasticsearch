/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.Arrays;

/**
 * A scalar quantizer for the residual that the first level IVF quantizer (OSQ or ASH) leaves behind on each vector.
 * <p>
 * Every dimension of every training residual is divided by that dimension's standard deviation and then pooled into a
 * single one dimensional collection, over which one Lloyd-Max codebook is fit. Sharing the codebook across dimensions
 * means the segment only stores {@code dimension + LEVELS} floats and the query side only needs one small table, while
 * pooling gives {@code N * dimension} samples per codebook entry instead of {@code N}. Per dimension scaling absorbs the
 * main difference between dimensions, which is their spread, so the shared codebook loses little against one fit per
 * dimension. The query side folds the scaling in algebraically, {@code <q, scale * decode(code)> == <q * scale, decode(code)>},
 * so scoring never touches the codebook itself.
 * <p>
 * The number of levels is fixed at {@link #LEVELS} (one byte per dimension) for now.
 */
public final class PooledLloydMaxQuantizer {

    /** Bits per encoded residual dimension. */
    public static final int RESIDUAL_BITS = 8;
    /** Number of codebook entries, {@code 2^RESIDUAL_BITS}. */
    public static final int LEVELS = 1 << RESIDUAL_BITS;
    /** Upper bound on the pooled residual values kept for fitting, which bounds the training memory. */
    public static final int MAX_TRAINING_VALUES = 4_000_000;
    /** Lower bound on the number of training vectors sampled, when the segment has that many. */
    public static final int MIN_TRAINING_VECTORS = 256;

    private static final int MAX_LLOYD_ITERATIONS = 100;
    private static final float MINIMUM_SCALE = 1e-12f;

    private final float[] scales;
    private final float[] codebook;
    private final float[] boundaries;

    PooledLloydMaxQuantizer(float[] scales, float[] codebook) {
        this.scales = scales;
        this.codebook = codebook;
        this.boundaries = new float[codebook.length - 1];
        for (int i = 0; i < boundaries.length; i++) {
            boundaries[i] = (codebook[i] + codebook[i + 1]) / 2f;
        }
    }

    /**
     * Number of training vectors to sample from a segment of {@code numberOfVectors} vectors of the given dimension so
     * that the pooled values stay within {@link #MAX_TRAINING_VALUES}.
     */
    public static int trainingSampleSize(int numberOfVectors, int dimension) {
        return Math.min(numberOfVectors, Math.max(MIN_TRAINING_VECTORS, MAX_TRAINING_VALUES / Math.max(1, dimension)));
    }

    public int dimension() {
        return scales.length;
    }

    public int levels() {
        return codebook.length;
    }

    public float[] scales() {
        return scales;
    }

    public float[] codebook() {
        return codebook;
    }

    /** Encodes a residual vector into one unsigned byte per dimension, dividing every dimension by its scale first. */
    public void encode(float[] residual, byte[] destination) {
        encode(residual, destination, 0);
    }

    /** Same as {@link #encode(float[], byte[])}, writing the codes at {@code destinationOffset}. */
    public void encode(float[] residual, byte[] destination, int destinationOffset) {
        for (int dimension = 0; dimension < residual.length; dimension++) {
            destination[destinationOffset + dimension] = (byte) encodeNormalized(residual[dimension] / scales[dimension]);
        }
    }

    /** Index of the nearest codebook entry to a scale normalized value. */
    int encodeNormalized(float normalizedValue) {
        int low = 0;
        int high = boundaries.length;
        while (low < high) {
            final int middle = (low + high) >>> 1;
            if (normalizedValue < boundaries[middle]) {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        return low;
    }

    /** Decodes one dimension of an encoded residual back into the residual's own scale. */
    public float decode(int dimension, byte code) {
        return codebook[code & 0xFF] * scales[dimension];
    }

    public void write(IndexOutput output) throws IOException {
        for (float scale : scales) {
            output.writeInt(Float.floatToIntBits(scale));
        }
        for (float entry : codebook) {
            output.writeInt(Float.floatToIntBits(entry));
        }
    }

    public static PooledLloydMaxQuantizer read(IndexInput input, int dimension) throws IOException {
        final float[] scales = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            scales[i] = Float.intBitsToFloat(input.readInt());
        }
        final float[] codebook = new float[LEVELS];
        for (int i = 0; i < LEVELS; i++) {
            codebook[i] = Float.intBitsToFloat(input.readInt());
        }
        return new PooledLloydMaxQuantizer(scales, codebook);
    }

    /**
     * Collects training residuals and fits the per dimension scales and the shared codebook. Residuals are copied, so a
     * caller may hand in a buffer it reuses. At most {@link #MAX_TRAINING_VALUES} values are kept; callers should sample
     * their vectors with {@link #trainingSampleSize} rather than relying on this bound.
     */
    public static final class Trainer {
        private final int dimension;
        private final int levels;
        private final double[] sums;
        private final double[] sumsOfSquares;
        private float[] values;
        private int numberOfValues;
        private long numberOfVectors;

        public Trainer(int dimension) {
            this(dimension, LEVELS);
        }

        Trainer(int dimension, int levels) {
            this.dimension = dimension;
            this.levels = levels;
            this.sums = new double[dimension];
            this.sumsOfSquares = new double[dimension];
            this.values = new float[Math.min(MAX_TRAINING_VALUES, Math.max(dimension, 1024))];
        }

        public int dimension() {
            return dimension;
        }

        public void add(float[] residual) {
            assert residual.length == dimension;
            for (int d = 0; d < dimension; d++) {
                sums[d] += residual[d];
                sumsOfSquares[d] += (double) residual[d] * residual[d];
            }
            numberOfVectors++;
            if (numberOfValues + dimension > MAX_TRAINING_VALUES) {
                return;
            }
            if (numberOfValues + dimension > values.length) {
                values = Arrays.copyOf(values, Math.min(MAX_TRAINING_VALUES, Math.max(values.length * 2, numberOfValues + dimension)));
            }
            System.arraycopy(residual, 0, values, numberOfValues, dimension);
            numberOfValues += dimension;
        }

        public PooledLloydMaxQuantizer fit() {
            if (numberOfVectors == 0) {
                throw new IllegalStateException("at least one training residual is required");
            }
            final float[] scales = new float[dimension];
            for (int d = 0; d < dimension; d++) {
                final double mean = sums[d] / numberOfVectors;
                final double variance = Math.max(0d, sumsOfSquares[d] / numberOfVectors - mean * mean);
                final float standardDeviation = (float) Math.sqrt(variance);
                scales[d] = standardDeviation < MINIMUM_SCALE ? 1f : standardDeviation;
            }
            final float[] normalized = new float[numberOfValues];
            for (int i = 0; i < numberOfValues; i++) {
                normalized[i] = values[i] / scales[i % dimension];
            }
            return new PooledLloydMaxQuantizer(scales, fitCodebook(normalized, levels));
        }
    }

    /**
     * Fits a one dimensional Lloyd-Max codebook on the given values, which are sorted in place. With sorted values and
     * their prefix sums every Lloyd iteration is a binary search per boundary plus one division per level, instead of a
     * pass over all values.
     *
     * @return the codebook in ascending order
     */
    static float[] fitCodebook(float[] values, int levels) {
        Arrays.sort(values);
        final int numberOfValues = values.length;
        final double[] prefixSums = new double[numberOfValues + 1];
        for (int i = 0; i < numberOfValues; i++) {
            prefixSums[i + 1] = prefixSums[i] + values[i];
        }
        final float[] centroids = initialCentroids(values, levels);
        final int[] regionStarts = new int[levels + 1];
        regionStarts[levels] = numberOfValues;
        for (int iteration = 0; iteration < MAX_LLOYD_ITERATIONS; iteration++) {
            boolean assignmentChanged = false;
            for (int level = 1; level < levels; level++) {
                final float boundary = (centroids[level - 1] + centroids[level]) / 2f;
                final int start = lowerBound(values, boundary);
                if (start != regionStarts[level]) {
                    assignmentChanged = true;
                    regionStarts[level] = start;
                }
            }
            if (iteration > 0 && assignmentChanged == false) {
                break;
            }
            for (int level = 0; level < levels; level++) {
                final int start = regionStarts[level];
                final int end = regionStarts[level + 1];
                if (end > start) {
                    centroids[level] = (float) ((prefixSums[end] - prefixSums[start]) / (end - start));
                }
            }
            Arrays.sort(centroids);
        }
        return centroids;
    }

    /**
     * Picks the initial centroids at evenly spaced quantiles of the sorted values. Heavily tied values make several
     * quantiles land on the same value and such duplicates would never separate in later iterations, so they are replaced
     * by values spread evenly between the smallest and the largest value.
     */
    private static float[] initialCentroids(float[] sortedValues, int levels) {
        final int numberOfValues = sortedValues.length;
        final float[] centroids = new float[levels];
        if (numberOfValues == 0) {
            return centroids;
        }
        int distinct = 0;
        for (int level = 0; level < levels; level++) {
            final float quantile = sortedValues[(int) (((level + 0.5d) * numberOfValues) / levels)];
            if (distinct == 0 || quantile != centroids[distinct - 1]) {
                centroids[distinct++] = quantile;
            }
        }
        if (distinct == levels) {
            return centroids;
        }
        final float smallest = sortedValues[0];
        final float largest = sortedValues[numberOfValues - 1];
        final int missing = levels - distinct;
        for (int i = 0; i < missing; i++) {
            centroids[distinct + i] = smallest == largest ? smallest : smallest + (largest - smallest) * (i + 1) / (missing + 1);
        }
        Arrays.sort(centroids);
        return centroids;
    }

    private static int lowerBound(float[] sortedValues, float target) {
        int low = 0;
        int high = sortedValues.length;
        while (low < high) {
            final int middle = (low + high) >>> 1;
            if (sortedValues[middle] < target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }
}

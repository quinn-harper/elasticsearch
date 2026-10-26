/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import java.util.Arrays;

/**
 * A scalar quantizer for the residual left over after the first level quantizer. Every dimension of every training
 * residual is divided by that dimension's standard deviation and then pooled into one collection of values, over which a
 * single shared Lloyd-Max codebook is fit. Sharing the codebook keeps the per segment overhead to a few floats, and the
 * per dimension scale absorbs the main difference between dimensions, which is their spread.
 */
final class PooledLloydMaxQuantizer {

    private static final int MAX_LLOYD_ITERATIONS = 100;
    private static final float MINIMUM_SCALE = 1e-12f;

    private final float[] codebook;
    private final float[] boundaries;
    private final float[] dimensionScales;

    PooledLloydMaxQuantizer(float[] codebook, float[] dimensionScales) {
        this.codebook = codebook;
        this.dimensionScales = dimensionScales;
        this.boundaries = new float[codebook.length - 1];
        for (int i = 0; i < boundaries.length; i++) {
            boundaries[i] = (codebook[i] + codebook[i + 1]) / 2f;
        }
    }

    /**
     * Fits the per dimension scales and the shared codebook on the given training residuals.
     *
     * @param trainingResiduals one residual vector per row
     * @param bitsPerValue number of bits per encoded dimension, giving {@code 2^bitsPerValue} codebook entries
     */
    static PooledLloydMaxQuantizer fit(float[][] trainingResiduals, int bitsPerValue) {
        if (trainingResiduals.length == 0) {
            throw new IllegalArgumentException("at least one training residual is required");
        }
        final int dimensions = trainingResiduals[0].length;
        final float[] scales = computeScales(trainingResiduals, dimensions);
        final float[] normalizedValues = new float[trainingResiduals.length * dimensions];
        int position = 0;
        for (float[] residual : trainingResiduals) {
            for (int dimension = 0; dimension < dimensions; dimension++) {
                normalizedValues[position++] = residual[dimension] / scales[dimension];
            }
        }
        return new PooledLloydMaxQuantizer(fitCodebook(normalizedValues, 1 << bitsPerValue), scales);
    }

    private static float[] computeScales(float[][] trainingResiduals, int dimensions) {
        final double[] sums = new double[dimensions];
        final double[] sumsOfSquares = new double[dimensions];
        for (float[] residual : trainingResiduals) {
            for (int dimension = 0; dimension < dimensions; dimension++) {
                sums[dimension] += residual[dimension];
                sumsOfSquares[dimension] += (double) residual[dimension] * residual[dimension];
            }
        }
        final float[] scales = new float[dimensions];
        final double numberOfVectors = trainingResiduals.length;
        for (int dimension = 0; dimension < dimensions; dimension++) {
            final double mean = sums[dimension] / numberOfVectors;
            final double variance = Math.max(0d, sumsOfSquares[dimension] / numberOfVectors - mean * mean);
            final float standardDeviation = (float) Math.sqrt(variance);
            scales[dimension] = standardDeviation < MINIMUM_SCALE ? 1f : standardDeviation;
        }
        return scales;
    }

    /**
     * Fits a one dimensional Lloyd-Max codebook. The values are sorted so each Lloyd iteration is a binary search per
     * boundary plus a prefix sum lookup per level, instead of a pass over all values.
     *
     * @param values the pooled normalized values, sorted in place
     * @param levels the number of codebook entries
     * @return the codebook in ascending order
     */
    static float[] fitCodebook(float[] values, int levels) {
        Arrays.sort(values);
        final int numberOfValues = values.length;
        final double[] prefixSums = new double[numberOfValues + 1];
        for (int i = 0; i < numberOfValues; i++) {
            prefixSums[i + 1] = prefixSums[i] + values[i];
        }

        final float[] centroids = new float[levels];
        for (int level = 0; level < levels; level++) {
            centroids[level] = values[(int) (((level + 0.5d) * numberOfValues) / levels)];
        }

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

    int levels() {
        return codebook.length;
    }

    int dimensions() {
        return dimensionScales.length;
    }

    float[] codebook() {
        return codebook;
    }

    float[] dimensionScales() {
        return dimensionScales;
    }

    /** Encodes one normalized value as the index of the nearest codebook entry. */
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

    /** Encodes a residual vector, dividing every dimension by its scale first. */
    void encode(float[] residual, int[] destination) {
        for (int dimension = 0; dimension < residual.length; dimension++) {
            destination[dimension] = encodeNormalized(residual[dimension] / dimensionScales[dimension]);
        }
    }

    float decode(int dimension, int code) {
        return codebook[code] * dimensionScales[dimension];
    }
}

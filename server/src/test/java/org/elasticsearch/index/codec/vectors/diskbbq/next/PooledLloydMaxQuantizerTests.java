/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.elasticsearch.test.ESTestCase;

import java.util.Arrays;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.lessThan;

public class PooledLloydMaxQuantizerTests extends ESTestCase {

    private static float[] gaussianValues(int numberOfValues, double standardDeviation) {
        final float[] values = new float[numberOfValues];
        for (int i = 0; i < numberOfValues; i++) {
            values[i] = (float) (random().nextGaussian() * standardDeviation);
        }
        return values;
    }

    private static double meanSquaredError(float[] values, float[] codebook) {
        final PooledLloydMaxQuantizer quantizer = new PooledLloydMaxQuantizer(codebook, new float[] { 1f });
        double sum = 0;
        for (float value : values) {
            final float decoded = quantizer.decode(0, quantizer.encodeNormalized(value));
            sum += (double) (value - decoded) * (value - decoded);
        }
        return sum / values.length;
    }

    public void testCodebookHasOneEntryPerLevelInAscendingOrder() {
        for (int bits = 1; bits <= 8; bits++) {
            final float[] codebook = PooledLloydMaxQuantizer.fitCodebook(gaussianValues(20_000, 1d), 1 << bits);
            assertThat(codebook.length, equalTo(1 << bits));
            for (int i = 1; i < codebook.length; i++) {
                assertTrue("bits=" + bits + " entry " + i, codebook[i - 1] <= codebook[i]);
            }
        }
    }

    public void testErrorShrinksWithEveryExtraBit() {
        final float[] values = gaussianValues(50_000, 1d);
        double previousError = Double.MAX_VALUE;
        for (int bits = 1; bits <= 8; bits++) {
            final float[] codebook = PooledLloydMaxQuantizer.fitCodebook(values.clone(), 1 << bits);
            final double error = meanSquaredError(values, codebook);
            assertThat("bits=" + bits, error, lessThan(previousError));
            previousError = error;
        }
    }

    public void testOneBitCodebookIsCloseToTheKnownGaussianOptimum() {
        final float[] codebook = PooledLloydMaxQuantizer.fitCodebook(gaussianValues(200_000, 1d), 2);
        final double expectedCentroid = Math.sqrt(2d / Math.PI);
        assertEquals(-expectedCentroid, codebook[0], 0.02d);
        assertEquals(expectedCentroid, codebook[1], 0.02d);
    }

    public void testEncodingPicksTheNearestCodebookEntry() {
        final float[][] residuals = new float[200][4];
        for (float[] residual : residuals) {
            for (int i = 0; i < residual.length; i++) {
                residual[i] = (float) random().nextGaussian();
            }
        }
        final PooledLloydMaxQuantizer quantizer = PooledLloydMaxQuantizer.fit(residuals, 3);
        final float[] probe = new float[] { -3f, -0.1f, 0.05f, 2.5f };
        final int[] codes = new int[probe.length];
        quantizer.encode(probe, codes);
        for (int dimension = 0; dimension < probe.length; dimension++) {
            final float normalized = probe[dimension] / quantizer.dimensionScales()[dimension];
            float bestDistance = Float.MAX_VALUE;
            for (float entry : quantizer.codebook()) {
                bestDistance = Math.min(bestDistance, Math.abs(normalized - entry));
            }
            final float chosenDistance = Math.abs(normalized - quantizer.codebook()[codes[dimension]]);
            assertEquals(bestDistance, chosenDistance, 1e-6f);
        }
    }

    public void testScaleNormalizationHelpsDimensionsThatAreMuchSmallerThanOthers() {
        final int dimensions = 8;
        final int numberOfVectors = 4_000;
        final double[] spreads = new double[dimensions];
        for (int dimension = 0; dimension < dimensions; dimension++) {
            spreads[dimension] = Math.pow(10d, dimension % 4 - 2);
        }
        final float[][] residuals = new float[numberOfVectors][dimensions];
        final float[] pooled = new float[numberOfVectors * dimensions];
        int position = 0;
        for (float[] residual : residuals) {
            for (int dimension = 0; dimension < dimensions; dimension++) {
                residual[dimension] = (float) (random().nextGaussian() * spreads[dimension]);
                pooled[position++] = residual[dimension];
            }
        }
        final int bits = 3;
        final PooledLloydMaxQuantizer scaled = PooledLloydMaxQuantizer.fit(residuals, bits);
        final float[] ones = new float[dimensions];
        Arrays.fill(ones, 1f);
        final PooledLloydMaxQuantizer unscaled = new PooledLloydMaxQuantizer(PooledLloydMaxQuantizer.fitCodebook(pooled.clone(), 1 << bits), ones);

        final double scaledRelativeError = meanRelativeError(scaled, residuals, spreads);
        final double unscaledRelativeError = meanRelativeError(unscaled, residuals, spreads);
        assertThat(scaledRelativeError, lessThan(unscaledRelativeError));
    }

    private static double meanRelativeError(PooledLloydMaxQuantizer quantizer, float[][] residuals, double[] spreads) {
        final int dimensions = spreads.length;
        final double[] squaredError = new double[dimensions];
        final int[] codes = new int[dimensions];
        for (float[] residual : residuals) {
            quantizer.encode(residual, codes);
            for (int dimension = 0; dimension < dimensions; dimension++) {
                final double difference = residual[dimension] - quantizer.decode(dimension, codes[dimension]);
                squaredError[dimension] += difference * difference;
            }
        }
        double relativeSum = 0;
        for (int dimension = 0; dimension < dimensions; dimension++) {
            relativeSum += squaredError[dimension] / residuals.length / (spreads[dimension] * spreads[dimension]);
        }
        return relativeSum / dimensions;
    }

    public void testDimensionWithNoVarianceKeepsAFiniteScale() {
        final float[][] residuals = new float[100][2];
        for (float[] residual : residuals) {
            residual[0] = 0f;
            residual[1] = (float) random().nextGaussian();
        }
        final PooledLloydMaxQuantizer quantizer = PooledLloydMaxQuantizer.fit(residuals, 2);
        assertThat(quantizer.dimensionScales()[0], equalTo(1f));
        final int[] codes = new int[2];
        quantizer.encode(new float[] { 0f, 0.5f }, codes);
        assertTrue(Float.isFinite(quantizer.decode(0, codes[0])));
    }
}

/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexInput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.IndexInput;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.lessThan;

/**
 * Ports the checks of the Python proof of concept ({@code python/test_pooled_lloyd_max.py}): the codebook converges on
 * Gaussian data, recovers well separated clusters, round trips with a small error that shrinks with more levels, and
 * per dimension scaling helps when dimensions differ in spread.
 */
public class PooledLloydMaxQuantizerTests extends ESTestCase {

    private static float[][] gaussian(Random random, int numberOfVectors, int dimension, float[] perDimensionScale) {
        float[][] vectors = new float[numberOfVectors][dimension];
        for (float[] vector : vectors) {
            for (int d = 0; d < dimension; d++) {
                vector[d] = (float) random.nextGaussian() * (perDimensionScale == null ? 1f : perDimensionScale[d]);
            }
        }
        return vectors;
    }

    private static PooledLloydMaxQuantizer fit(float[][] vectors, int levels) {
        PooledLloydMaxQuantizer.Trainer trainer = new PooledLloydMaxQuantizer.Trainer(vectors[0].length, levels);
        for (float[] vector : vectors) {
            trainer.add(vector);
        }
        return trainer.fit();
    }

    private static double roundTripMeanSquaredError(PooledLloydMaxQuantizer quantizer, float[][] vectors) {
        byte[] codes = new byte[vectors[0].length];
        double error = 0;
        for (float[] vector : vectors) {
            quantizer.encode(vector, codes);
            for (int d = 0; d < vector.length; d++) {
                double difference = vector[d] - quantizer.decode(d, codes[d]);
                error += difference * difference;
            }
        }
        return error / ((double) vectors.length * vectors[0].length);
    }

    private static double variance(float[][] vectors) {
        double sum = 0;
        double sumOfSquares = 0;
        long count = 0;
        for (float[] vector : vectors) {
            for (float value : vector) {
                sum += value;
                sumOfSquares += (double) value * value;
                count++;
            }
        }
        double mean = sum / count;
        return sumOfSquares / count - mean * mean;
    }

    public void testFitConvergesOnGaussianData() {
        float[] values = new float[200_000];
        Random random = random();
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) random.nextGaussian();
        }
        float[] sorted = values.clone();
        Arrays.sort(sorted);
        float[] codebook = PooledLloydMaxQuantizer.fitCodebook(values, 16);
        assertThat(codebook.length, equalTo(16));
        for (int i = 1; i < codebook.length; i++) {
            assertThat("codebook must be sorted and distinct", codebook[i], greaterThan(codebook[i - 1]));
        }
        assertThat(codebook[0], greaterThan(sorted[0]));
        assertThat(codebook[codebook.length - 1], lessThan(sorted[sorted.length - 1]));
    }

    public void testFitRecoversWellSeparatedClusters() {
        float[] trueCenters = { -5f, 0f, 5f };
        float[] values = new float[3000];
        Random random = random();
        for (int i = 0; i < values.length; i++) {
            values[i] = trueCenters[i % 3] + 0.1f * (float) random.nextGaussian();
        }
        float[] codebook = PooledLloydMaxQuantizer.fitCodebook(values, 3);
        for (int i = 0; i < 3; i++) {
            assertThat((double) codebook[i], closeTo(trueCenters[i], 0.05));
        }
    }

    public void testEncodeDecodeRoundTripReducesError() {
        float[][] vectors = gaussian(random(), 2000, 25, null);
        PooledLloydMaxQuantizer quantizer = fit(vectors, 64);
        assertThat(roundTripMeanSquaredError(quantizer, vectors), lessThan(0.05 * variance(vectors)));
    }

    public void testMoreLevelsReduceError() {
        float[][] vectors = gaussian(random(), 2000, 25, null);
        double previous = Double.MAX_VALUE;
        for (int levels : new int[] { 4, 16, 64, 256 }) {
            double error = roundTripMeanSquaredError(fit(vectors, levels), vectors);
            assertThat("levels " + levels, error, lessThan(previous));
            previous = error;
        }
    }

    public void testDefaultLevelsIsOneBytePerDimension() {
        float[][] vectors = gaussian(random(), 500, 8, null);
        PooledLloydMaxQuantizer quantizer = fit(vectors, PooledLloydMaxQuantizer.LEVELS);
        assertThat(quantizer.levels(), equalTo(256));
        assertThat(quantizer.dimension(), equalTo(8));
        byte[] codes = new byte[8];
        quantizer.encode(vectors[0], codes);
        for (int d = 0; d < 8; d++) {
            assertThat((double) quantizer.decode(d, codes[d]), closeTo(vectors[0][d], 0.1));
        }
    }

    public void testScaleNormalizationTracksPerDimensionSpread() {
        int dimension = 16;
        float[] scale = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            scale[d] = 1f + 4f * d / (dimension - 1);
        }
        float[][] vectors = gaussian(random(), 5000, dimension, scale);
        PooledLloydMaxQuantizer quantizer = fit(vectors, 16);
        assertThat(quantizer.scales().length, equalTo(dimension));
        assertThat(quantizer.scales()[dimension - 1], greaterThan(quantizer.scales()[0]));
        for (int d = 0; d < dimension; d++) {
            assertThat((double) quantizer.scales()[d], closeTo(scale[d], 0.2 * scale[d]));
        }
    }

    public void testScaleNormalizationBeatsPlainPoolingOnHeteroscedasticDimensions() {
        int dimension = 16;
        float[] scale = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            scale[d] = 1f + 7f * d / (dimension - 1);
        }
        float[][] vectors = gaussian(random(), 20_000, dimension, scale);
        PooledLloydMaxQuantizer scaled = fit(vectors, 8);
        float[] unitScales = new float[dimension];
        Arrays.fill(unitScales, 1f);
        float[] pooled = new float[vectors.length * dimension];
        int position = 0;
        for (float[] vector : vectors) {
            for (float value : vector) {
                pooled[position++] = value;
            }
        }
        PooledLloydMaxQuantizer plain = new PooledLloydMaxQuantizer(unitScales, PooledLloydMaxQuantizer.fitCodebook(pooled, 8));
        assertThat(roundTripMeanSquaredError(scaled, vectors), lessThan(roundTripMeanSquaredError(plain, vectors)));
    }

    public void testDegenerateInputs() {
        float[] values = { 0f, 0f, 1f, 1f };
        float[] codebook = PooledLloydMaxQuantizer.fitCodebook(values, 5);
        assertThat(codebook.length, equalTo(5));
        for (int i = 1; i < codebook.length; i++) {
            assertThat(codebook[i], greaterThan(codebook[i - 1] - 1e-6f));
        }

        float[] allEqual = new float[100];
        Arrays.fill(allEqual, 3f);
        float[] constantCodebook = PooledLloydMaxQuantizer.fitCodebook(allEqual, 4);
        for (float entry : constantCodebook) {
            assertThat((double) entry, closeTo(3f, 1e-6));
        }

        float[][] vectors = gaussian(random(), 200, 3, null);
        for (float[] vector : vectors) {
            vector[1] = 0f;
        }
        PooledLloydMaxQuantizer quantizer = fit(vectors, 16);
        assertThat(quantizer.scales()[1], equalTo(1f));
        byte[] codes = new byte[3];
        quantizer.encode(vectors[0], codes);
        assertThat((double) quantizer.decode(1, codes[1]), closeTo(0d, 0.5));
    }

    public void testWriteReadRoundTrip() throws IOException {
        float[][] vectors = gaussian(random(), 1000, 12, null);
        PooledLloydMaxQuantizer quantizer = fit(vectors, PooledLloydMaxQuantizer.LEVELS);
        ByteBuffersDataOutput dataOutput = new ByteBuffersDataOutput();
        try (ByteBuffersIndexOutput output = new ByteBuffersIndexOutput(dataOutput, "test", "test")) {
            quantizer.write(output);
        }
        try (IndexInput input = new ByteBuffersIndexInput(dataOutput.toDataInput(), "test")) {
            PooledLloydMaxQuantizer read = PooledLloydMaxQuantizer.read(input, 12);
            assertArrayEquals(quantizer.scales(), read.scales(), 0f);
            assertArrayEquals(quantizer.codebook(), read.codebook(), 0f);
            byte[] expected = new byte[12];
            byte[] actual = new byte[12];
            for (float[] vector : vectors) {
                quantizer.encode(vector, expected);
                read.encode(vector, actual);
                assertArrayEquals(expected, actual);
            }
        }
    }

    public void testTrainingSampleSize() {
        assertThat(PooledLloydMaxQuantizer.trainingSampleSize(10, 128), equalTo(10));
        assertThat(
            PooledLloydMaxQuantizer.trainingSampleSize(1_000_000, 1024),
            equalTo(PooledLloydMaxQuantizer.MAX_TRAINING_VALUES / 1024)
        );
        assertThat(
            PooledLloydMaxQuantizer.trainingSampleSize(1_000_000, PooledLloydMaxQuantizer.MAX_TRAINING_VALUES),
            equalTo(PooledLloydMaxQuantizer.MIN_TRAINING_VECTORS)
        );
    }
}

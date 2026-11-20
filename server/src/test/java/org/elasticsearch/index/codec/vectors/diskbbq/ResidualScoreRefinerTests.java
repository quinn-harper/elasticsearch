/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.test.ESTestCase;

import static org.hamcrest.Matchers.closeTo;

public class ResidualScoreRefinerTests extends ESTestCase {

    private static PooledLloydMaxQuantizer quantizer(int dimension) {
        PooledLloydMaxQuantizer.Trainer trainer = new PooledLloydMaxQuantizer.Trainer(dimension);
        for (int i = 0; i < 2000; i++) {
            float[] residual = new float[dimension];
            for (int d = 0; d < dimension; d++) {
                residual[d] = (float) random().nextGaussian() * (1 + d);
            }
            trainer.add(residual);
        }
        return trainer.fit();
    }

    private static float transform(VectorSimilarityFunction similarityFunction, float estimate) {
        return switch (similarityFunction) {
            case EUCLIDEAN -> 1f / (1f + Math.max(0f, estimate));
            case MAXIMUM_INNER_PRODUCT -> VectorUtil.scaleMaxInnerProductScore(estimate);
            case DOT_PRODUCT, COSINE -> Math.max((1f + estimate) / 2f, 0f);
        };
    }

    /** {@code refineScore} must equal transforming the corrected dot product or squared distance directly. */
    public void testRefineScoreMatchesDirectTransform() {
        PooledLloydMaxQuantizer quantizer = quantizer(8);
        for (VectorSimilarityFunction similarityFunction : VectorSimilarityFunction.values()) {
            ResidualScoreRefiner refiner = new ResidualScoreRefiner(similarityFunction, quantizer);
            for (int i = 0; i < 100; i++) {
                float delta = randomFloatBetween(-0.5f, 0.5f, true);
                float estimate;
                float corrected;
                if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
                    estimate = randomFloatBetween(0.5f, 50f, true);
                    corrected = estimate - 2f * delta;
                } else if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
                    estimate = randomFloatBetween(-5f, 5f, true);
                    corrected = estimate + delta;
                } else {
                    estimate = randomFloatBetween(-0.9f, 0.9f, true);
                    corrected = estimate + delta;
                }
                float coarse = transform(similarityFunction, estimate);
                float expected = transform(similarityFunction, corrected);
                assertThat(similarityFunction.toString(), (double) refiner.refineScore(coarse, delta), closeTo(expected, 1e-4));
            }
        }
    }

    public void testDeltaIsDotProductWithDecodedResidual() {
        int dimension = 12;
        PooledLloydMaxQuantizer quantizer = quantizer(dimension);
        ResidualScoreRefiner refiner = new ResidualScoreRefiner(VectorSimilarityFunction.DOT_PRODUCT, quantizer);
        float[] centeredQuery = new float[dimension];
        float[] residual = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            centeredQuery[d] = randomFloatBetween(-1f, 1f, true);
            residual[d] = (float) random().nextGaussian() * (1 + d);
        }
        byte[] codes = new byte[dimension + 3];
        quantizer.encode(residual, codes, 3);
        double expected = 0;
        for (int d = 0; d < dimension; d++) {
            expected += centeredQuery[d] * quantizer.decode(d, codes[3 + d]);
        }
        refiner.setCenteredQuery(centeredQuery);
        assertThat((double) refiner.delta(codes, 3), closeTo(expected, 1e-4));

        float[] scaled = new float[dimension];
        refiner.scaleQuery(centeredQuery, scaled);
        refiner.setScaledQuery(scaled);
        assertThat((double) refiner.delta(codes, 3), closeTo(expected, 1e-4));
    }

    public void testRefineScoreNeverProducesNaN() {
        PooledLloydMaxQuantizer quantizer = quantizer(4);
        for (VectorSimilarityFunction similarityFunction : VectorSimilarityFunction.values()) {
            ResidualScoreRefiner refiner = new ResidualScoreRefiner(similarityFunction, quantizer);
            for (float coarse : new float[] { 0f, Float.MIN_VALUE, 1e-3f, 0.5f, 1f, 100f }) {
                for (float delta : new float[] { -1000f, -1f, 0f, 1f, 1000f }) {
                    float refined = refiner.refineScore(coarse, delta);
                    assertFalse(similarityFunction + " " + coarse + " " + delta, Float.isNaN(refined));
                }
            }
        }
    }
}

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

/**
 * Turns the residual codes stored next to a vector's first level codes into a correction of that vector's coarse score.
 * <p>
 * Both first level quantizers estimate the similarity as {@code <q_c, xhat_c> + C}, where {@code q_c} is the centered
 * query, {@code xhat_c} the reconstruction of the centered vector and {@code C} exact stored terms. The residual
 * {@code r = x_c - xhat_c} is the only thing missing, so {@code delta = <q_c, r>} added to the estimated dot product
 * (or {@code 2 * delta} subtracted from the estimated squared distance) gives the refined score. The decoded residual is
 * {@code scale[i] * codebook[code[i]]}, which is folded into the query once per centroid as {@code qs = q_c * scale}.
 */
public final class ResidualScoreRefiner {

    private final VectorSimilarityFunction similarityFunction;
    private final float[] codebook;
    private final float[] scales;
    private final float[] ownScaledQuery;
    private float[] scaledQuery;

    public ResidualScoreRefiner(VectorSimilarityFunction similarityFunction, PooledLloydMaxQuantizer quantizer) {
        this.similarityFunction = similarityFunction;
        this.codebook = quantizer.codebook();
        this.scales = quantizer.scales();
        this.ownScaledQuery = new float[quantizer.dimension()];
        this.scaledQuery = ownScaledQuery;
    }

    public int dimension() {
        return ownScaledQuery.length;
    }

    /** Sets the centered query the coarse estimator is built around; the per dimension scales are folded in here. */
    public void setCenteredQuery(float[] centeredQuery) {
        scaleQuery(centeredQuery, ownScaledQuery);
        scaledQuery = ownScaledQuery;
    }

    /** Fills {@code destination} with {@code centeredQuery * scales} so callers can cache it per centroid. */
    public void scaleQuery(float[] centeredQuery, float[] destination) {
        for (int i = 0; i < ownScaledQuery.length; i++) {
            destination[i] = centeredQuery[i] * scales[i];
        }
    }

    /** Uses an already scaled query, as produced by {@link #scaleQuery}. The array is referenced, not copied. */
    public void setScaledQuery(float[] scaled) {
        assert scaled.length == ownScaledQuery.length;
        scaledQuery = scaled;
    }

    /** The residual dot product {@code <q_c, r>} for the codes at {@code offset}. */
    public float delta(byte[] codes, int offset) {
        float delta = 0f;
        for (int i = 0; i < scaledQuery.length; i++) {
            delta = Math.fma(scaledQuery[i], codebook[codes[offset + i] & 0xFF], delta);
        }
        return delta;
    }

    /**
     * Applies {@code delta} to a score that already went through the similarity transform, by inverting the transform
     * the IVF scorers use, adjusting the underlying dot product or squared distance, and transforming again. Used where
     * the coarse score comes out of a fused bulk scorer that does not expose the dot product.
     */
    public float refineScore(float coarseScore, float delta) {
        return switch (similarityFunction) {
            case EUCLIDEAN -> {
                final float distance = coarseScore <= 0f ? Float.MAX_VALUE : 1f / coarseScore - 1f;
                yield 1f / (1f + Math.max(0f, distance - 2f * delta));
            }
            case MAXIMUM_INNER_PRODUCT -> {
                final float dot = coarseScore >= 1f ? coarseScore - 1f : 1f - 1f / Math.max(coarseScore, Float.MIN_NORMAL);
                yield VectorUtil.scaleMaxInnerProductScore(dot + delta);
            }
            case DOT_PRODUCT, COSINE -> {
                final float dot = 2f * coarseScore - 1f;
                yield Math.max((1f + dot + delta) / 2f, 0f);
            }
        };
    }

    /** Applies {@code delta} to an estimated dot product, before any similarity transform. */
    public static float refineDotProduct(float dotProduct, float delta) {
        return dotProduct + delta;
    }
}

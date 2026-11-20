/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import java.io.IOException;

/**
 * Implemented by rescoring vector values that store their data grouped by index structure, so a whole set of candidates can
 * be scored in storage order instead of one random read per candidate in the order they were found.
 */
public interface OrderedRescoreVectorValues {

    /**
     * Scores the given vector ordinals against the target, reading their data in storage order.
     *
     * @param target the query vector
     * @param ordinals the vector ordinals to score, in any order
     * @param count the number of leading entries of {@code ordinals} to score
     * @param scoresOut receives the score of {@code ordinals[i]} at index {@code i}
     */
    void scoreOrdinals(float[] target, int[] ordinals, int count, float[] scoresOut) throws IOException;
}

/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.FloatVectorValues;

import java.io.IOException;

/**
 * Implemented by {@link org.apache.lucene.codecs.KnnVectorsReader} implementations that store compact residual codes
 * alongside their index, so the rescoring phase can score candidates without reading the raw float vectors.
 * This is a separate interface, rather than an override of {@code getFloatVectorValues}, because the raw vectors are
 * still needed unchanged by merges, exact search and scripts.
 */
public interface QuantizedRescoringReader {

    /**
     * Returns vector values that share the ordinals and doc ids of the raw vectors but whose scorers estimate the
     * similarity from the stored residual codes, or {@code null} when this segment has no such codes for the field.
     */
    FloatVectorValues getQuantizedRescoreValues(String field) throws IOException;
}

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
 * A vector reader whose segments may carry a quantized copy of each vector for rescoring. The rescore query asks for a
 * {@link QuantizedRescorer} per segment and falls back to the raw vectors when it gets {@code null}, so segments with and
 * without a copy can live in one index.
 */
public interface QuantizedRescoringReader {

    /**
     * Returns a rescorer over the field's quantized copy for {@code target}, or {@code null} when this segment stores no
     * copy for the field.
     */
    QuantizedRescorer quantizedRescorer(String field, float[] target) throws IOException;
}

/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

/**
 * Implemented by {@link org.apache.lucene.codecs.KnnVectorsReader} implementations whose approximate search already
 * refines candidate scores with residual codes stored in the posting lists. The query layer uses it to skip the
 * raw vector rescoring pass for such segments, since the scores they return are already the refined ones.
 */
public interface ResidualRescoringReader {

    /** Whether the given field stores residual codes in this segment, so its search results need no raw vector rescore. */
    boolean hasResidualCodes(String field);
}

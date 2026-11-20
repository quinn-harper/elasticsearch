/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.search.VectorScorer;

import java.io.IOException;

/**
 * Rescores candidates of one segment from its quantized rescore copy instead of the raw vectors. The scorer iterates the
 * field's vectors by document like a raw vector rescorer does, so the rescore query can use either one unchanged. The
 * record methods address the copy by record position instead, so a caller with many candidates can sort them by record
 * and read the copy front to back rather than in document order, which scatters the reads over the whole copy.
 */
public interface QuantizedRescorer {

    /** Scores the field's vectors for the query this rescorer was created for. */
    VectorScorer scorer();

    /** The position in the copy of the record written for the vector at {@code ord}. */
    int recordIndex(int ord) throws IOException;

    /** Hints that the record at {@code recordIndex} is about to be read. */
    void prefetchRecord(int recordIndex) throws IOException;

    /** Scores the vector whose record is at {@code recordIndex}, as returned by {@link #recordIndex(int)}. */
    float scoreRecord(int recordIndex) throws IOException;
}

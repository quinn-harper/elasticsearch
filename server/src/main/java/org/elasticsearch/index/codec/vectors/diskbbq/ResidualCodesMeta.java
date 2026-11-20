/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

/**
 * Describes the residual codes a segment stores for one field: the quantizer that decodes them and the region of the
 * residual file that holds them. The residual codes live in their own file, laid out in posting list order with
 * {@code dimension} bytes per vector, so that the posting lists stay as compact as without reranking and the cache
 * can keep them resident while the residual bytes come and go. Each posting list header records where its vectors'
 * codes start inside the field's region.
 * <p>
 * Written at the end of the field meta behind one byte holding the residual bits per dimension; zero means the field
 * has no residual codes and nothing else follows.
 *
 * @param quantizer decodes the stored codes
 * @param offset    start of the field's codes in the residual file
 * @param length    number of bytes the field's codes take in the residual file
 */
public record ResidualCodesMeta(PooledLloydMaxQuantizer quantizer, long offset, long length) {

    /** Writes {@code meta}, or the marker for a field without residual codes when it is {@code null}. */
    public static void write(IndexOutput metaOutput, ResidualCodesMeta meta) throws IOException {
        if (meta == null) {
            metaOutput.writeByte((byte) 0);
            return;
        }
        metaOutput.writeByte((byte) PooledLloydMaxQuantizer.RESIDUAL_BITS);
        meta.quantizer.write(metaOutput);
        metaOutput.writeLong(meta.offset);
        metaOutput.writeLong(meta.length);
    }

    /** Reads what {@link #write} wrote, returning {@code null} for a field without residual codes. */
    public static ResidualCodesMeta read(IndexInput input, int dimension) throws IOException {
        final int residualBits = input.readByte();
        if (residualBits == 0) {
            return null;
        }
        if (residualBits != PooledLloydMaxQuantizer.RESIDUAL_BITS) {
            throw new IOException(
                "unsupported residual bits [" + residualBits + "], expected [" + PooledLloydMaxQuantizer.RESIDUAL_BITS + "]"
            );
        }
        final PooledLloydMaxQuantizer quantizer = PooledLloydMaxQuantizer.read(input, dimension);
        final long offset = input.readLong();
        final long length = input.readLong();
        return new ResidualCodesMeta(quantizer, offset, length);
    }

    /** The field's region of the residual file; posting list headers hold offsets relative to its start. */
    public IndexInput slice(IndexInput residualFile) throws IOException {
        return residualFile.slice("residualCodes", offset, length);
    }
}

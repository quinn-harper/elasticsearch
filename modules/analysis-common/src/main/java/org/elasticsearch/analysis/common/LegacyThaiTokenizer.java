/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.analysis.common;

import org.apache.lucene.analysis.th.ThaiTokenizer;

/**
 * A {@link ThaiTokenizer} that only cuts its internal buffer at line and paragraph separators, as Lucene
 * 10.5 and earlier did. Lucene 10.6 also accepts whitespace (apache/lucene#16727), which moves token
 * boundaries in text with no line break for 1024 or more characters. Indices created before Lucene 10.6
 * keep the old cut points so they do not need to be reindexed.
 */
final class LegacyThaiTokenizer extends ThaiTokenizer {

    @Override
    protected boolean isSafeEnd(char ch) {
        switch (ch) {
            case 0x000D:
            case 0x000A:
            case 0x0085:
            case 0x2028:
            case 0x2029:
                return true;
            default:
                return false;
        }
    }
}

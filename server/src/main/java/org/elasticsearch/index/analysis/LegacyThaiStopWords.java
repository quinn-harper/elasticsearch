/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.analysis;

import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.WordlistLoader;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;

/**
 * The Thai stop word list shipped with Lucene 10.5 and earlier. Lucene 10.6 curated the list
 * (apache/lucene#16718), which changes the tokens produced for the {@code _thai_} stop word name and for
 * the built-in {@code thai} analyzer. Indices created before Lucene 10.6 keep this list so they do not need to
 * be reindexed.
 */
public final class LegacyThaiStopWords {

    private static final String RESOURCE_NAME = "thai_stopwords_lucene_10_5.txt";
    private static final String COMMENT_PREFIX = "#";

    private LegacyThaiStopWords() {}

    private static class Holder {
        static final CharArraySet STOP_SET;

        static {
            try (InputStream stream = LegacyThaiStopWords.class.getResourceAsStream(RESOURCE_NAME)) {
                if (stream == null) {
                    throw new IllegalStateException("Missing resource [" + RESOURCE_NAME + "]");
                }
                STOP_SET = CharArraySet.unmodifiableSet(WordlistLoader.getWordSet(stream, COMMENT_PREFIX));
            } catch (IOException e) {
                throw new UncheckedIOException("Unable to load the legacy Thai stop word set", e);
            }
        }
    }

    /**
     * Returns an unmodifiable set containing the Lucene 10.5 default Thai stop words.
     */
    public static CharArraySet getStopSet() {
        return Holder.STOP_SET;
    }
}

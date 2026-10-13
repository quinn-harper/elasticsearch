/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.analysis.common;

import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.StopwordAnalyzerBase;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.DecimalDigitFilter;
import org.elasticsearch.index.analysis.LegacyThaiStopWords;

/**
 * The Thai analyzer as built by Lucene 10.5 and earlier: tokenizer, lowercase, decimal digit and stop
 * filters. Lucene 10.6 added a char filter, {@code thai_normalization} and {@code thai_repeat}, and curated
 * the default stop words (apache/lucene#16717, #16718, #16720). Indices created before Lucene 10.6 keep this
 * chain for the {@code thai} analyzer so they do not need to be reindexed.
 */
public final class LegacyThaiAnalyzer extends StopwordAnalyzerBase {

    public LegacyThaiAnalyzer() {
        this(LegacyThaiStopWords.getStopSet());
    }

    public LegacyThaiAnalyzer(CharArraySet stopwords) {
        super(stopwords);
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        final Tokenizer source = new LegacyThaiTokenizer();
        TokenStream result = new LowerCaseFilter(source);
        result = new DecimalDigitFilter(result);
        result = new StopFilter(result, stopwords);
        return new TokenStreamComponents(source, result);
    }

    @Override
    protected TokenStream normalize(String fieldName, TokenStream in) {
        TokenStream result = new LowerCaseFilter(in);
        result = new DecimalDigitFilter(result);
        return result;
    }
}

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
import org.apache.lucene.analysis.StopwordAnalyzerBase;
import org.apache.lucene.analysis.th.ThaiAnalyzer;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.env.Environment;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.IndexVersions;
import org.elasticsearch.index.analysis.AbstractIndexAnalyzerProvider;
import org.elasticsearch.index.analysis.Analysis;
import org.elasticsearch.index.analysis.LegacyThaiStopWords;

public class ThaiAnalyzerProvider extends AbstractIndexAnalyzerProvider<StopwordAnalyzerBase> {

    private final StopwordAnalyzerBase analyzer;
    private final Object sharingKey;

    ThaiAnalyzerProvider(IndexSettings indexSettings, Environment env, String name, Settings settings) {
        super(name);
        boolean modernThai = indexSettings.getIndexVersionCreated().onOrAfter(IndexVersions.UPGRADE_TO_LUCENE_10_6_0);
        CharArraySet defaultStopWords = modernThai ? ThaiAnalyzer.getDefaultStopSet() : LegacyThaiStopWords.getStopSet();
        CharArraySet stopWords = Analysis.parseStopWords(indexSettings.getIndexVersionCreated(), env, settings, defaultStopWords);
        // since Lucene 10.6 this analyzer normalizes Thai orthography, expands Maiyamok and uses a curated stop list,
        // older indices keep the previous behavior
        analyzer = modernThai ? new ThaiAnalyzer(stopWords) : new LegacyThaiAnalyzer(stopWords);
        this.sharingKey = new Key(Analysis.stableStopWords(settings, stopWords), modernThai);
    }

    @Override
    public StopwordAnalyzerBase get() {
        return this.analyzer;
    }

    @Override
    public Object sharingKey() {
        return sharingKey;
    }

    private record Key(Analysis.StableCharArraySet stopWords, boolean modernThai) {}
}

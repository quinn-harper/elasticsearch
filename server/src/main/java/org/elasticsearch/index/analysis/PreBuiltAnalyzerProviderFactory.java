/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.analysis;

import org.apache.lucene.analysis.Analyzer;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.env.Environment;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.indices.analysis.PreBuiltAnalyzers;
import org.elasticsearch.indices.analysis.PreBuiltCacheFactory;

import java.io.Closeable;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class PreBuiltAnalyzerProviderFactory extends PreConfiguredAnalysisComponent<AnalyzerProvider<?>> implements Closeable {

    private final Function<IndexVersion, Analyzer> create;
    private final PreBuiltAnalyzerProvider current;

    /**
     * This constructor only exists to expose analyzers defined in {@link PreBuiltAnalyzers} as {@link PreBuiltAnalyzerProviderFactory}.
     */
    PreBuiltAnalyzerProviderFactory(String name, PreBuiltAnalyzers preBuiltAnalyzer) {
        super(name, new PreBuiltAnalyzersDelegateCache(name, preBuiltAnalyzer));
        this.create = preBuiltAnalyzer::getAnalyzer;
        Analyzer analyzer = preBuiltAnalyzer.getAnalyzer(IndexVersion.current());
        current = new PreBuiltAnalyzerProvider(name, AnalyzerScope.INDICES, analyzer);
    }

    public PreBuiltAnalyzerProviderFactory(String name, PreBuiltCacheFactory.CachingStrategy cache, Supplier<Analyzer> create) {
        super(name, cache);
        this.create = version -> create.get();
        Analyzer analyzer = create.get();
        this.current = new PreBuiltAnalyzerProvider(name, AnalyzerScope.INDICES, analyzer);
    }

    private PreBuiltAnalyzerProviderFactory(Function<IndexVersion, Analyzer> create, String name, PreBuiltCacheFactory.CachingStrategy cache) {
        super(name, cache);
        this.create = create;
        this.current = new PreBuiltAnalyzerProvider(name, AnalyzerScope.INDICES, create.apply(IndexVersion.current()));
    }

    /**
     * Creates a factory for an analyzer whose behavior depends on the version the index was created with. With
     * {@link PreBuiltCacheFactory.CachingStrategy#LUCENE} the analyzer for a version is cached per Lucene version, so {@code create} may only
     * vary its result on boundaries that coincide with a Lucene version change.
     */
    public static PreBuiltAnalyzerProviderFactory indexVersion(
        String name,
        PreBuiltCacheFactory.CachingStrategy cache,
        Function<IndexVersion, Analyzer> create
    ) {
        return new PreBuiltAnalyzerProviderFactory(create, name, cache);
    }

    @Override
    public AnalyzerProvider<?> get(IndexSettings indexSettings, Environment environment, String name, Settings settings)
        throws IOException {
        IndexVersion versionCreated = indexSettings.getIndexVersionCreated();
        if (IndexVersion.current().equals(versionCreated) == false) {
            return super.get(indexSettings, environment, name, settings);
        } else {
            return current;
        }
    }

    @Override
    protected AnalyzerProvider<?> create(IndexVersion version) {
        assert IndexVersion.current().equals(version) == false;
        Analyzer analyzer = create.apply(version);
        return new PreBuiltAnalyzerProvider(getName(), AnalyzerScope.INDICES, analyzer);
    }

    @Override
    public void close() throws IOException {
        List<Closeable> closeables = cache.values().stream().<Closeable>map(AnalyzerProvider::get).toList();
        IOUtils.close(current.get(), () -> IOUtils.close(closeables));
    }

    /**
     *  A special cache that closes the gap between PreBuiltAnalyzers and PreBuiltAnalyzerProviderFactory.
     *
     *  This can be removed when all analyzers have been moved away from PreBuiltAnalyzers to
     *  PreBuiltAnalyzerProviderFactory either in server or analysis-common.
     */
    static class PreBuiltAnalyzersDelegateCache implements PreBuiltCacheFactory.PreBuiltCache<AnalyzerProvider<?>> {

        private final String name;
        private final PreBuiltAnalyzers preBuiltAnalyzer;

        private PreBuiltAnalyzersDelegateCache(String name, PreBuiltAnalyzers preBuiltAnalyzer) {
            this.name = name;
            this.preBuiltAnalyzer = preBuiltAnalyzer;
        }

        @Override
        public AnalyzerProvider<?> get(IndexVersion version) {
            return new PreBuiltAnalyzerProvider(name, AnalyzerScope.INDICES, preBuiltAnalyzer.getAnalyzer(version));
        }

        @Override
        public void put(IndexVersion version, AnalyzerProvider<?> analyzerProvider) {
            // No need to put, because we delegate in get() directly to PreBuiltAnalyzers which already caches.
        }

        @Override
        public Collection<AnalyzerProvider<?>> values() {
            return preBuiltAnalyzer.getCache()
                .values()
                .stream()
                // Wrap the analyzer instance in a PreBuiltAnalyzerProvider, this is what PreBuiltAnalyzerProviderFactory#close expects
                // (other caches are not directly caching analyzers, but analyzer provider instead)
                .<AnalyzerProvider<?>>map(analyzer -> new PreBuiltAnalyzerProvider(name, AnalyzerScope.INDICES, analyzer))
                .toList();
        }

    }
}

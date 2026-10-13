/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.analysis.common;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.env.Environment;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.IndexVersions;
import org.elasticsearch.index.analysis.AnalysisTestsHelper;
import org.elasticsearch.index.analysis.LegacyThaiStopWords;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.test.ESTokenStreamTestCase;
import org.elasticsearch.test.IndexSettingsModule;
import org.elasticsearch.test.index.IndexVersionUtils;

import java.io.IOException;
import java.util.Arrays;

import static org.apache.lucene.tests.analysis.BaseTokenStreamTestCase.assertAnalyzesTo;

/**
 * Lucene 10.6 changed the output of {@code ThaiAnalyzer}, its default stop list, and {@code ThaiTokenizer}
 * (apache/lucene#16717, #16718, #16720, #16727). Indices created before
 * {@link IndexVersions#UPGRADE_TO_LUCENE_10_6_0} must keep their previous token streams for the {@code thai}
 * analyzer, the {@code thai} tokenizer and the {@code _thai_} stop word name so they do not need to be
 * reindexed. Newer indices get the Lucene 10.6 behavior.
 */
public class ThaiAnalyzerBwcTests extends ESTokenStreamTestCase {

    private static IndexVersion randomLegacyVersion() {
        return IndexVersionUtils.randomPreviousCompatibleWriteVersion(IndexVersions.UPGRADE_TO_LUCENE_10_6_0);
    }

    private static IndexVersion randomModernVersion() {
        return IndexVersionUtils.randomVersionOnOrAfter(IndexVersions.UPGRADE_TO_LUCENE_10_6_0);
    }

    /**
     * Named {@code analyzer: thai} ({@link ThaiAnalyzerProvider}). Through 10.5 that was tokenizer, lowercase,
     * decimal_digit, stop. Lucene 10.6 added a char filter and normalization (double Sara E becomes a single
     * one), expansion of Maiyamok ({@code เร็วๆ} gives two {@code เร็ว} tokens) and a curated stop list
     * (content words such as {@code ไป} are no longer stopped).
     */
    public void testNamedThaiAnalyzerOnLegacyIndex() throws IOException {
        assertPreLucene106ThaiTokens(namedThaiAnalyzer(randomLegacyVersion()));
    }

    public void testNamedThaiAnalyzerOnModernIndex() throws IOException {
        assertLucene106ThaiTokens(namedThaiAnalyzer(randomModernVersion()));
    }

    /**
     * Prebuilt {@code thai} is a second registration of the analyzer. Mappings that only set
     * {@code "analyzer": "thai"} use it, so it must match the named analyzer for the same index version.
     */
    public void testPrebuiltThaiAnalyzerOnLegacyIndex() throws IOException {
        assertPreLucene106ThaiTokens(analysisFor(randomLegacyVersion(), Settings.builder()).indexAnalyzers.get("thai"));
    }

    public void testPrebuiltThaiAnalyzerOnModernIndex() throws IOException {
        assertLucene106ThaiTokens(analysisFor(randomModernVersion(), Settings.builder()).indexAnalyzers.get("thai"));
    }

    /**
     * {@code stopwords: _thai_} follows the index version. The test uses only {@code tokenizer: thai} plus a stop
     * filter, so it isolates the named stop set from the other analyzer changes.
     */
    public void testThaiStopwordNameOnLegacyIndex() throws IOException {
        Analyzer analyzer = thaiStopsAnalyzer(randomLegacyVersion());
        assertAnalyzesTo(analyzer, "ไป", new String[] {});
        assertAnalyzesTo(analyzer, "ที่ผ่านมา", new String[] {});
    }

    public void testThaiStopwordNameOnModernIndex() throws IOException {
        Analyzer analyzer = thaiStopsAnalyzer(randomModernVersion());
        assertAnalyzesTo(analyzer, "ไป", new String[] { "ไป" });
    }

    /**
     * Named {@code tokenizer: thai} ({@link ThaiTokenizerFactory}). Through 10.5 only newlines were a safe cut
     * for the 1024 character buffer, so a word that straddled offset 1024 after a space was sliced into
     * fragments. Lucene 10.6 also accepts whitespace.
     */
    public void testNamedThaiTokenizerOnLegacyIndex() throws IOException {
        Settings.Builder settings = Settings.builder()
            .put("index.analysis.tokenizer.my_thai.type", "thai")
            .put("index.analysis.analyzer.thai_only.tokenizer", "my_thai");
        assertPreLucene106ThaiTokenizer(analysisFor(randomLegacyVersion(), settings).indexAnalyzers.get("thai_only"));
    }

    public void testNamedThaiTokenizerOnModernIndex() throws IOException {
        Settings.Builder settings = Settings.builder()
            .put("index.analysis.tokenizer.my_thai.type", "thai")
            .put("index.analysis.analyzer.thai_only.tokenizer", "my_thai");
        assertLucene106ThaiTokenizer(analysisFor(randomModernVersion(), settings).indexAnalyzers.get("thai_only"));
    }

    /**
     * Preconfigured {@code thai} tokenizer is the no-settings twin of {@link ThaiTokenizerFactory}, used by
     * {@code _analyze} and by mappings that name the tokenizer without declaring a factory.
     */
    public void testPreconfiguredThaiTokenizerOnLegacyIndex() throws IOException {
        Settings.Builder settings = Settings.builder().put("index.analysis.analyzer.thai_only.tokenizer", "thai");
        assertPreLucene106ThaiTokenizer(analysisFor(randomLegacyVersion(), settings).indexAnalyzers.get("thai_only"));
    }

    public void testPreconfiguredThaiTokenizerOnModernIndex() throws IOException {
        Settings.Builder settings = Settings.builder().put("index.analysis.analyzer.thai_only.tokenizer", "thai");
        assertLucene106ThaiTokenizer(analysisFor(randomModernVersion(), settings).indexAnalyzers.get("thai_only"));
    }

    /**
     * Lucene 10.6 {@code ThaiAnalyzer} also normalizes at query time ({@code normalize}, used for wildcard,
     * prefix and fuzzy terms) with the char filter and {@code ThaiNormalizationFilter}. Legacy indices must
     * leave doubled Sara E untouched there, as they do at index time.
     */
    public void testNormalizeOnLegacyIndex() throws IOException {
        IndexVersion version = randomLegacyVersion();
        for (Analyzer analyzer : new Analyzer[] {
            namedThaiAnalyzer(version),
            analysisFor(version, Settings.builder()).indexAnalyzers.get("thai") }) {
            assertEquals(new BytesRef("เเมว"), analyzer.normalize("field", "เเมว"));
        }
    }

    public void testNormalizeOnModernIndex() throws IOException {
        IndexVersion version = randomModernVersion();
        for (Analyzer analyzer : new Analyzer[] {
            namedThaiAnalyzer(version),
            analysisFor(version, Settings.builder()).indexAnalyzers.get("thai") }) {
            assertEquals(new BytesRef("แมว"), analyzer.normalize("field", "เเมว"));
        }
    }

    /**
     * Analyzers and tokenizers are shared across indices when their sharing keys are equal, so indices on
     * different sides of the Lucene 10.6 boundary must never produce equal keys.
     */
    public void testSharingKeysDifferAcrossLucene106() {
        IndexVersion legacy = randomLegacyVersion();
        IndexVersion modern = randomModernVersion();
        assertNotEquals(thaiTokenizerFactory(legacy).sharingKey(), thaiTokenizerFactory(modern).sharingKey());
        assertEquals(thaiTokenizerFactory(legacy).sharingKey(), thaiTokenizerFactory(legacy).sharingKey());
        assertEquals(thaiTokenizerFactory(modern).sharingKey(), thaiTokenizerFactory(modern).sharingKey());
        assertNotEquals(thaiAnalyzerProvider(legacy).sharingKey(), thaiAnalyzerProvider(modern).sharingKey());
        assertEquals(thaiAnalyzerProvider(legacy).sharingKey(), thaiAnalyzerProvider(legacy).sharingKey());
        assertEquals(thaiAnalyzerProvider(modern).sharingKey(), thaiAnalyzerProvider(modern).sharingKey());
    }

    /**
     * The legacy list has the 115 entries of Lucene 10.5. Entries Lucene 10.6 removed as over-filtering, such as
     * {@code ไป}, are present, and entries it added are absent.
     */
    public void testLegacyThaiStopWords() {
        CharArraySet stopWords = LegacyThaiStopWords.getStopSet();
        assertEquals(115, stopWords.size());
        assertTrue(stopWords.contains("ไป"));
        assertTrue(stopWords.contains("ผ่าน"));
        assertTrue(stopWords.contains("มา"));
        assertFalse(stopWords.contains("สำหรับ"));
        assertFalse(stopWords.contains("ทำให้"));
    }

    private static IndexSettings indexSettings(IndexVersion version) {
        Settings settings = ESTestCase.indexSettings(1, 1).put(IndexMetadata.SETTING_VERSION_CREATED, version).build();
        return IndexSettingsModule.newIndexSettings("index", settings);
    }

    private Environment environment() {
        return new Environment(Settings.builder().put(Environment.PATH_HOME_SETTING.getKey(), createTempDir().toString()).build(), null);
    }

    private Analyzer namedThaiAnalyzer(IndexVersion version) {
        return thaiAnalyzerProvider(version).get();
    }

    private ThaiAnalyzerProvider thaiAnalyzerProvider(IndexVersion version) {
        return new ThaiAnalyzerProvider(indexSettings(version), environment(), "my-analyzer", Settings.EMPTY);
    }

    private ThaiTokenizerFactory thaiTokenizerFactory(IndexVersion version) {
        return new ThaiTokenizerFactory(indexSettings(version), environment(), "my-tokenizer", Settings.EMPTY);
    }

    private ESTestCase.TestAnalysis analysisFor(IndexVersion version, Settings.Builder analysisSettings) throws IOException {
        Settings settings = analysisSettings.put(Environment.PATH_HOME_SETTING.getKey(), createTempDir().toString())
            .put(IndexMetadata.SETTING_VERSION_CREATED, version)
            .build();
        return AnalysisTestsHelper.createTestAnalysisFromSettings(settings, new CommonAnalysisPlugin());
    }

    private Analyzer thaiStopsAnalyzer(IndexVersion version) throws IOException {
        Settings.Builder settings = Settings.builder()
            .put("index.analysis.filter.thai_stops.type", "stop")
            .put("index.analysis.filter.thai_stops.stopwords", "_thai_")
            .put("index.analysis.analyzer.thai_stops.tokenizer", "thai")
            .put("index.analysis.analyzer.thai_stops.filter", "thai_stops");
        return analysisFor(version, settings).indexAnalyzers.get("thai_stops");
    }

    /**
     * Pre-10.6 {@code ThaiAnalyzer} tokens: #16717 ({@code เเ} stays), #16720 ({@code ๆ} is a leftover token) and
     * #16718 ({@code ไป}, {@code ที่ผ่านมา} are stopped).
     */
    private static void assertPreLucene106ThaiTokens(Analyzer analyzer) throws IOException {
        assertAnalyzesTo(analyzer, "เเปลก", new String[] { "เเปลก" });
        assertAnalyzesTo(analyzer, "เเมว", new String[] { "เเมว" });
        assertAnalyzesTo(analyzer, "เร็วๆ", new String[] { "เร็ว", "ๆ" });
        assertAnalyzesTo(analyzer, "ไป", new String[] {});
        assertAnalyzesTo(analyzer, "ที่ผ่านมา", new String[] {});
    }

    /**
     * Lucene 10.6 {@code ThaiAnalyzer} tokens for the same inputs as {@link #assertPreLucene106ThaiTokens}.
     */
    private static void assertLucene106ThaiTokens(Analyzer analyzer) throws IOException {
        assertAnalyzesTo(analyzer, "เเปลก", new String[] { "แปลก" });
        assertAnalyzesTo(analyzer, "เเมว", new String[] { "แมว" });
        assertAnalyzesTo(analyzer, "เร็วๆ", new String[] { "เร็ว", "เร็ว" });
        assertAnalyzesTo(analyzer, "ไป", new String[] { "ไป" });
        assertAnalyzesTo(analyzer, "ที่ผ่านมา", new String[] { "ผ่าน", "มา" });
    }

    private static final String LONG_BUFFER_TEXT = "ก".repeat(1021) + " แมว";

    /**
     * Pre-10.6 {@code ThaiTokenizer} long-buffer tokens: {@code แมว} straddles the 1024 character buffer and is
     * cut into {@code แม} and {@code ว}.
     */
    private static void assertPreLucene106ThaiTokenizer(Analyzer analyzer) throws IOException {
        String[] expected = new String[513];
        Arrays.fill(expected, 0, 510, "กก");
        expected[510] = "ก";
        expected[511] = "แม";
        expected[512] = "ว";
        assertAnalyzesTo(analyzer, LONG_BUFFER_TEXT, expected);
    }

    /**
     * Lucene 10.6 treats the space before {@code แมว} as a safe cut, so the word stays in one token.
     */
    private static void assertLucene106ThaiTokenizer(Analyzer analyzer) throws IOException {
        String[] expected = new String[512];
        Arrays.fill(expected, 0, 510, "กก");
        expected[510] = "ก";
        expected[511] = "แมว";
        assertAnalyzesTo(analyzer, LONG_BUFFER_TEXT, expected);
    }
}

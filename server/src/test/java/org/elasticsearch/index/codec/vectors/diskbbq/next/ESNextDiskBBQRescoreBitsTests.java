/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantEncoding;
import org.elasticsearch.index.codec.vectors.diskbbq.QuantizedRescoringReader;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.search.vectors.RescoreKnnVectorQuery;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import static org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION;
import static org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat.defaultFlatThreshold;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;

/**
 * Checks that segments written with {@code rescoreBits} store residual codes that reproduce the raw vector scores
 * closely, that the accuracy depends on the number of bits, and that the codes survive merges.
 */
public class ESNextDiskBBQRescoreBitsTests extends ESTestCase {

    private static final String VECTOR_FIELD = "vector";
    private static final String MATCH_FIELD = "match";
    private static final int DIMENSIONS = 64;
    private static final int VECTORS_PER_CLUSTER = 128;

    private static Codec codec(int rescoreBits) {
        return TestUtil.alwaysKnnVectorsFormat(
            new ESNextDiskBBQVectorsFormat(
                QuantEncoding.ONE_BIT_4BIT_QUERY,
                VECTORS_PER_CLUSTER,
                ESNextDiskBBQVectorsFormat.MIN_CENTROIDS_PER_PARENT_CLUSTER,
                DenseVectorFieldMapper.ElementType.FLOAT,
                false,
                null,
                1,
                false,
                DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
                defaultFlatThreshold(VECTORS_PER_CLUSTER),
                null,
                IvfFlushConfigSource.empty(),
                IvfMergeConfigResolver.useCodecDefault(),
                false,
                rescoreBits
            )
        );
    }

    private static float[] randomUnitVector() {
        final float[] vector = new float[DIMENSIONS];
        double squaredNorm = 0;
        for (int i = 0; i < DIMENSIONS; i++) {
            vector[i] = (float) random().nextGaussian();
            squaredNorm += (double) vector[i] * vector[i];
        }
        final float norm = (float) Math.sqrt(squaredNorm);
        for (int i = 0; i < DIMENSIONS; i++) {
            vector[i] /= norm;
        }
        return vector;
    }

    private static void addVectors(IndexWriter writer, int numberOfVectors, VectorSimilarityFunction similarity) throws IOException {
        for (int i = 0; i < numberOfVectors; i++) {
            final Document document = new Document();
            document.add(new StringField(MATCH_FIELD, "all", Field.Store.NO));
            document.add(new KnnFloatVectorField(VECTOR_FIELD, randomUnitVector(), similarity));
            writer.addDocument(document);
        }
    }

    private static QuantizedRescoringReader quantizedReader(LeafReader leafReader) {
        final SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(leafReader);
        assertThat(segmentReader, notNullValue());
        final KnnVectorsReader fieldsReader = segmentReader.getVectorReader();
        assertTrue(fieldsReader instanceof PerFieldKnnVectorsFormat.FieldsReader);
        final KnnVectorsReader fieldReader = ((PerFieldKnnVectorsFormat.FieldsReader) fieldsReader).getFieldReader(VECTOR_FIELD);
        assertTrue(fieldReader instanceof QuantizedRescoringReader);
        return (QuantizedRescoringReader) fieldReader;
    }

    /** The mean absolute difference between the quantized score and the exact score over every vector of the segment. */
    private static double meanScoreError(LeafReader leafReader, VectorSimilarityFunction similarity, float[] query) throws IOException {
        final FloatVectorValues quantized = quantizedReader(leafReader).getQuantizedRescoreValues(VECTOR_FIELD);
        assertThat(quantized, notNullValue());
        final FloatVectorValues raw = leafReader.getFloatVectorValues(VECTOR_FIELD);
        assertThat(quantized.size(), equalTo(raw.size()));

        final VectorScorer scorer = quantized.rescorer(query);
        final KnnVectorValues.DocIndexIterator rawIterator = raw.iterator();
        double totalError = 0;
        int scored = 0;
        for (int doc = rawIterator.nextDoc(); doc != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS; doc = rawIterator.nextDoc()) {
            assertThat(scorer.iterator().advance(doc), equalTo(doc));
            final float exactScore = similarity.compare(query, raw.vectorValue(rawIterator.index()));
            totalError += Math.abs(scorer.score() - exactScore);
            scored++;
        }
        assertThat(scored, equalTo(raw.size()));
        return totalError / scored;
    }

    private static double meanScoreErrorAfterIndexing(int rescoreBits, VectorSimilarityFunction similarity, int numberOfVectors)
        throws IOException {
        try (Directory directory = newDirectory()) {
            final IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer()).setCodec(codec(rescoreBits));
            try (IndexWriter writer = new IndexWriter(directory, config)) {
                addVectors(writer, numberOfVectors, similarity);
                writer.commit();
                writer.forceMerge(1);
            }
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                assertThat(reader.leaves().size(), equalTo(1));
                return meanScoreError(reader.leaves().get(0).reader(), similarity, randomUnitVector());
            }
        }
    }

    public void testQuantizedScoresAreMoreAccurateWithMoreBits() throws IOException {
        final VectorSimilarityFunction similarity = randomFrom(VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.DOT_PRODUCT);
        final int numberOfVectors = randomIntBetween(600, 1_200);

        final double oneBitError = meanScoreErrorAfterIndexing(1, similarity, numberOfVectors);
        final double fourBitError = meanScoreErrorAfterIndexing(4, similarity, numberOfVectors);
        final double eightBitError = meanScoreErrorAfterIndexing(8, similarity, numberOfVectors);

        assertThat(fourBitError, lessThan(oneBitError));
        assertThat(eightBitError, lessThan(fourBitError));
        assertThat("eight bits should be nearly exact", eightBitError, lessThan(5e-3));
    }

    public void testSectionIsRebuiltWhenSegmentsAreMerged() throws IOException {
        final VectorSimilarityFunction similarity = VectorSimilarityFunction.EUCLIDEAN;
        try (Directory directory = newDirectory()) {
            final IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer()).setCodec(codec(8));
            try (IndexWriter writer = new IndexWriter(directory, config)) {
                for (int segment = 0; segment < 3; segment++) {
                    addVectors(writer, randomIntBetween(400, 700), similarity);
                    writer.commit();
                }
                try (DirectoryReader beforeMerge = DirectoryReader.open(writer)) {
                    assertThat(beforeMerge.leaves().size(), greaterThanOrEqualTo(2));
                    for (LeafReaderContext leaf : beforeMerge.leaves()) {
                        assertThat(meanScoreError(leaf.reader(), similarity, randomUnitVector()), lessThan(5e-3));
                    }
                }
                writer.forceMerge(1);
            }
            try (DirectoryReader afterMerge = DirectoryReader.open(directory)) {
                assertThat(afterMerge.leaves().size(), equalTo(1));
                assertThat(meanScoreError(afterMerge.leaves().get(0).reader(), similarity, randomUnitVector()), lessThan(5e-3));
            }
        }
    }

    public void testNoQuantizedValuesWhenRescoreBitsIsZero() throws IOException {
        try (Directory directory = newDirectory()) {
            final IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer()).setCodec(codec(0));
            try (IndexWriter writer = new IndexWriter(directory, config)) {
                addVectors(writer, 500, VectorSimilarityFunction.EUCLIDEAN);
            }
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                for (LeafReaderContext leaf : reader.leaves()) {
                    assertThat(quantizedReader(leaf.reader()).getQuantizedRescoreValues(VECTOR_FIELD), nullValue());
                }
            }
        }
    }

    public void testQuantizedValuesKeepRawOrdinalsAndDocIds() throws IOException {
        try (Directory directory = newDirectory()) {
            final IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer()).setCodec(codec(4));
            try (IndexWriter writer = new IndexWriter(directory, config)) {
                for (int i = 0; i < 600; i++) {
                    final Document document = new Document();
                    document.add(new StringField(MATCH_FIELD, "all", Field.Store.NO));
                    if (i % 3 != 0) {
                        document.add(new KnnFloatVectorField(VECTOR_FIELD, randomUnitVector(), VectorSimilarityFunction.EUCLIDEAN));
                    }
                    writer.addDocument(document);
                }
                writer.forceMerge(1);
            }
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leaf = reader.leaves().get(0).reader();
                final FloatVectorValues raw = leaf.getFloatVectorValues(VECTOR_FIELD);
                final FloatVectorValues quantized = quantizedReader(leaf).getQuantizedRescoreValues(VECTOR_FIELD);
                assertThat(quantized.size(), equalTo(raw.size()));
                assertThat(quantized.dimension(), equalTo(raw.dimension()));
                for (int ord = 0; ord < raw.size(); ord++) {
                    assertThat(quantized.ordToDoc(ord), equalTo(raw.ordToDoc(ord)));
                }
            }
        }
    }

    public void testRescoreQueryUsesQuantizedCodesUnlessOptedOut() throws IOException {
        final VectorSimilarityFunction similarity = VectorSimilarityFunction.EUCLIDEAN;
        final int k = 10;
        try (Directory directory = newDirectory()) {
            final IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer()).setCodec(codec(8));
            try (IndexWriter writer = new IndexWriter(directory, config)) {
                addVectors(writer, 800, similarity);
                writer.forceMerge(1);
            }
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final IndexSearcher searcher = new IndexSearcher(reader);
                final float[] query = randomUnitVector();
                final TermQuery everything = new TermQuery(new Term(MATCH_FIELD, "all"));

                final ScoreDoc[] raw = searcher.search(RescoreKnnVectorQuery.fromInnerQuery(VECTOR_FIELD, query, k, 800, everything, false), k)
                    .scoreDocs;
                final ScoreDoc[] quantized = searcher.search(
                    RescoreKnnVectorQuery.fromInnerQuery(VECTOR_FIELD, query, k, 800, everything, true),
                    k
                ).scoreDocs;

                assertThat(raw.length, equalTo(k));
                assertThat(quantized.length, equalTo(k));
                final Set<Integer> rawDocs = new HashSet<>();
                for (ScoreDoc scoreDoc : raw) {
                    rawDocs.add(scoreDoc.doc);
                }
                int overlap = 0;
                boolean anyScoreDiffers = false;
                for (ScoreDoc scoreDoc : quantized) {
                    if (rawDocs.contains(scoreDoc.doc)) {
                        overlap++;
                    }
                }
                for (int i = 0; i < k; i++) {
                    anyScoreDiffers |= raw[i].score != quantized[i].score;
                }
                assertThat("eight bit quantized rescoring should find nearly the same neighbours", overlap, greaterThanOrEqualTo(k - 2));
                assertTrue("quantized scores are estimates and should not equal the exact scores", anyScoreDiffers);
            }
        }
    }

    public void testFormatRejectsUnsupportedRescoreBitsCombinations() {
        expectThrows(IllegalArgumentException.class, () -> formatWith(9, false, DenseVectorFieldMapper.ElementType.FLOAT));
        expectThrows(IllegalArgumentException.class, () -> formatWith(-1, false, DenseVectorFieldMapper.ElementType.FLOAT));
        expectThrows(IllegalArgumentException.class, () -> formatWith(4, true, DenseVectorFieldMapper.ElementType.FLOAT));
        expectThrows(IllegalArgumentException.class, () -> formatWith(4, false, DenseVectorFieldMapper.ElementType.BYTE));
        expectThrows(IllegalArgumentException.class, () -> formatWith(4, false, DenseVectorFieldMapper.ElementType.BFLOAT16));
    }

    private static ESNextDiskBBQVectorsFormat formatWith(
        int rescoreBits,
        boolean doPrecondition,
        DenseVectorFieldMapper.ElementType elementType
    ) {
        return new ESNextDiskBBQVectorsFormat(
            QuantEncoding.ONE_BIT_4BIT_QUERY,
            VECTORS_PER_CLUSTER,
            ESNextDiskBBQVectorsFormat.MIN_CENTROIDS_PER_PARENT_CLUSTER,
            elementType,
            false,
            null,
            1,
            doPrecondition,
            DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
            defaultFlatThreshold(VECTORS_PER_CLUSTER),
            null,
            IvfFlushConfigSource.empty(),
            IvfMergeConfigResolver.useCodecDefault(),
            false,
            rescoreBits
        );
    }
}

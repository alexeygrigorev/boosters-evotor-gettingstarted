package evotor;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.alexeygrigorev.dstools.cv.Split;
import com.alexeygrigorev.dstools.data.Dataset;
import com.alexeygrigorev.dstools.data.Datasets;
import com.alexeygrigorev.dstools.text.CountVectorizer;
import com.alexeygrigorev.dstools.text.Stopwords;
import com.alexeygrigorev.dstools.text.TextUtils;
import com.alexeygrigorev.dstools.text.TruncatedSVD;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableMap;

import joinery.DataFrame;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import smile.data.SparseDataset;

public class GettingStarted {

    private static final Logger LOGGER = LoggerFactory.getLogger(GettingStarted.class);

    public static void main(String[] args) throws Exception {
        DataFrame<Object> dfTrain = DataFrame.readCsv("data/evo_train.csv");
        List<Long> target = dfTrain.cast(Long.class).col("GROUP_ID");

        BiMap<Long, Integer> targetMapping = buildTargetMapping(target);
        double[] y = target.stream().mapToDouble(v -> targetMapping.get(v)).toArray();

        List<String> text = dfTrain.cast(String.class).col("NAME");

        LOGGER.info("tokenizing the train set...");
        List<List<String>> textTokenized = tokenizeBatch(text);

        LOGGER.info("vectorizing the train set...");
        CountVectorizer vectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(10)
                .withL2Normalization()
                .fit(textTokenized);


        LOGGER.info("svd-ing the train set...");
        SparseDataset Xtrain = vectorizer.transfrom(textTokenized);
        TruncatedSVD svd = new TruncatedSVD(150, true);
        double[][] XTrainLsa = svd.fitTransform(Xtrain);

        Dataset trainDataset = Datasets.of(XTrainLsa, y);
        Split trainValSplit = trainDataset.trainTestSplit(0.1);


        LOGGER.info("xgboost!");
        DMatrix dtrain = XgbUtils.wrapData(trainValSplit.getTrain());
        DMatrix dval = XgbUtils.wrapData(trainValSplit.getTest());
        Map<String, DMatrix> watchlist = ImmutableMap.of("train", dtrain, "val", dval);

        Map<String, Object> params = XgbUtils.defaultParams();
        params.put("objective", "multi:softprob");
        params.put("eval_metric", "merror");
        params.put("num_class", targetMapping.size());
        params.put("subsample", 0.7);
        params.put("colsample_bytree", 0.7);

        int nrounds = 35;
        Booster model = XGBoost.train(dtrain, params, nrounds, watchlist, null, null);

        LOGGER.info("reading the test set...");
        DataFrame<Object> dfTest = DataFrame.readCsv("data/evo_test.csv");
        List<String> textTest = dfTest.cast(String.class).col("NAME");

        LOGGER.info("tokenizing the test set...");
        List<List<String>> testTextTokenized = tokenizeBatch(textTest);

        LOGGER.info("vectorizing the test set...");
        SparseDataset Xtest = vectorizer.transfrom(testTextTokenized);

        LOGGER.info("svd-ing the test set...");
        double[][] XTestLsa = svd.transform(Xtest);

        LOGGER.info("predicting the result...");
        Dataset testDataset = Datasets.of(XTestLsa);
        int[] result = XgbUtils.predictBestClass(model, testDataset);

        LOGGER.info("creating the submission...");
        List<Object> prediction = invertGroupId(targetMapping, result);

        DataFrame<Object> results = dfTest.retain("id").add("GROUP_ID", prediction);
        results.writeCsv("submission.csv");

        LOGGER.info("done");
    }

    private static List<Object> invertGroupId(BiMap<Long, Integer> targetMapping, int[] result) {
        BiMap<Integer, Long> inverseTargetMapping = targetMapping.inverse();
        List<Object> prediction = Arrays.stream(result)
                .mapToObj(i -> inverseTargetMapping.get(i))
                .collect(Collectors.toList());
        return prediction;
    }

    private static BiMap<Long, Integer> buildTargetMapping(List<Long> target) {
        List<Long> groups = target.stream().distinct().sorted().collect(Collectors.toList());

        BiMap<Long, Integer> targetMapping = HashBiMap.create();
        int c = 0;
        for (Long group : groups) {
            targetMapping.put(group, c);
            c++;
        }

        return targetMapping;
    }

    private static List<List<String>> tokenizeBatch(List<String> text) {
        return text.stream().map(name -> tokenizeTitle(name)).collect(Collectors.toList());
    }

    private static List<String> tokenizeTitle(String title) {
        List<String> tokens = TextUtils.tokenizeRussian(title);
        tokens = TextUtils.removeStopwords(tokens, Stopwords.RU_STOPWORDS);
        return TextUtils.ngrams(tokens, 1, 3);
    }
}

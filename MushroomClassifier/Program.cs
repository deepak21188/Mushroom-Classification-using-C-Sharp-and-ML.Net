using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using MushroomClassifier.DataModels;
using static Microsoft.ML.TrainCatalogBase;


namespace MushroomClassifier
{
    class Program
    {
        static readonly string _dataFilePath = Path.Combine(@"C:\Users\dkumar\Documents\Deepak_ML\MushroomClassifier", "Data", "mushrooms.csv");
        static void Main(string[] args)
        {
            //Creating MLContxet model to be shared accross model building, validation and prediction process
            MLContext mlContext = new MLContext();

            //Loading the data from csv files
            //Splitting the dataset into train/test sets
            TrainTestData mushroomTrainTestData = LoadData(mlContext, testDataFraction: 0.25);

            //Creating data transformation pipeline which transforma the data a form acceptable by model
            //Returns an object of type IEstimator<ITransformer>
            var pipeline = ProcessData(mlContext);

            //passing the transformation pipeline and training dataset to crossvalidate and build the model
            //returns the model object of type ITransformer 
            var trainedModel = BuildAndTrain(mlContext, pipeline, mushroomTrainTestData.TrainSet);

            //Sample datainput for predicrtion
            var mushroomInput1 = new MushroomModelInput
            {
                cap_shape = "x",
                cap_surface="s",
                cap_color= "n",
                bruises= "t",
                odor= "p",
                gill_attachment="f",
                gill_spacing="c",
                gill_size="n",
                gill_color="k",
                stalk_shape="e",
                stalk_root="e",
                stalk_surface_above_ring="s",
                stalk_surface_below_ring="s",
                stalk_color_above_ring="w",
                stalk_color_below_ring="w",
                veil_type="p",
                veil_color="w",
                ring_number="o",
                ring_type = "p",
                spore_print_color = "k",
                population="s",
                habitat="u"
            };

            //Sample datainput for predicrtion
            var mushroomInput2 = new MushroomModelInput
            {
                cap_shape = "b",
                cap_surface = "y",
                cap_color = "y",
                bruises = "t",
                odor = "l",
                gill_attachment = "f",
                gill_spacing = "c",
                gill_size = "b",
                gill_color = "k",
                stalk_shape = "e",
                stalk_root = "c",
                stalk_surface_above_ring = "s",
                stalk_surface_below_ring = "s",
                stalk_color_above_ring = "w",
                stalk_color_below_ring = "w",
                veil_type = "p",
                veil_color = "w",
                ring_number = "o",
                ring_type = "p",
                spore_print_color = "n",
                population = "s",
                habitat = "m"
            };
            
            //passing trained model and sample input data to make single prediction 
            var result = PredictSingleResult(mlContext, trainedModel, mushroomInput2);
            
            Console.WriteLine("================================= Single Prediction Result ===============================");
            // Evaluate(mlContext, pipeline, trainedModel,  mushroomTrainTestData.TestSet);
            Console.WriteLine($"Predicted Result: {result.Label}");

            Console.ReadKey();




        }

        public static TrainTestData LoadData(MLContext mlContext, double testDataFraction)
        {
            IDataView mushroomDataView = mlContext.Data.LoadFromTextFile<MushroomModelInput>(_dataFilePath, hasHeader: true, separatorChar: ',', allowSparse: false);

            TrainTestData mushroomTrainTestData = mlContext.Data.TrainTestSplit(mushroomDataView, testFraction: testDataFraction);

            return mushroomTrainTestData;
        }

        public static IEstimator<ITransformer> ProcessData(MLContext mlContext)
        {
            var pipeline =      mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(MushroomModelInput.mClass))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "cap_shape", outputColumnName: "cap_shapeFeaturized"))                
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "cap_surface", outputColumnName: "cap_surfaceFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "cap_color", outputColumnName: "cap_colorFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "bruises", outputColumnName: "bruisesFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "odor", outputColumnName: "odorFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "gill_attachment", outputColumnName: "gill_attachmentFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "gill_spacing", outputColumnName: "gill_spacingFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "gill_size", outputColumnName: "gill_sizeFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "gill_color", outputColumnName: "gill_colorFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "stalk_shape", outputColumnName: "stalk_shapeFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "stalk_root", outputColumnName: "stalk_rootFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "stalk_surface_above_ring", outputColumnName: "stalk_surface_above_ringFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "stalk_surface_below_ring", outputColumnName: "stalk_surface_below_ringFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "stalk_color_above_ring", outputColumnName: "stalk_color_above_ringFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "stalk_color_below_ring", outputColumnName: "stalk_color_below_ringFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "veil_type", outputColumnName: "veil_typeFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "veil_color", outputColumnName: "veil_colorFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "ring_number", outputColumnName: "ring_numberFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "spore_print_color", outputColumnName: "spore_print_colorFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "population", outputColumnName: "populationFeaturized"))
                                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "habitat", outputColumnName: "habitatFeaturized"))
                                .Append(mlContext.Transforms.Concatenate(outputColumnName:"Features", inputColumnNames:new string[]{ "cap_shapeFeaturized", "cap_surfaceFeaturized"
                                                                                   , "cap_colorFeaturized", "bruisesFeaturized", "odorFeaturized", "gill_attachmentFeaturized"
                                                                                   , "gill_spacingFeaturized", "gill_sizeFeaturized", "gill_colorFeaturized", "stalk_shapeFeaturized"
                                                                                   , "stalk_rootFeaturized", "stalk_surface_above_ringFeaturized", "stalk_surface_below_ringFeaturized"
                                                                                   , "stalk_color_above_ringFeaturized", "stalk_color_below_ringFeaturized", "veil_typeFeaturized", "veil_colorFeaturized"
                                                                                   , "ring_numberFeaturized", "spore_print_colorFeaturized", "populationFeaturized", "habitatFeaturized" }));


            

              return pipeline;

        }

        public static ITransformer BuildAndTrain(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView trainDataView)
        {
          
         //   PeekDataViewInConsole(mlContext, trainDataView, pipeline, 2);
                      

            Console.WriteLine("=============== Create and Train the Model ===============");
            var averagedPerceptronBinaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10);
            
            var trainPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer))
                                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //var transformer = pipeline.Fit(trainDataView);
            //var transformedData = transformer.Transform(trainDataView);

            var crossValResults = mlContext.MulticlassClassification.CrossValidate(data: trainDataView, estimator: trainPipeline, numberOfFolds: 6, labelColumnName: "Label");

            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
          

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
          

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
          

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###} ");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###} ");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###} ");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###} ");
            Console.WriteLine($"*************************************************************************************************************");

            
           
            var model = trainPipeline.Fit(trainDataView);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;

        }

        public static MushroomModelPrediction PredictSingleResult(MLContext mlContext, ITransformer model, MushroomModelInput input) {

            var predictEngine = mlContext.Model.CreatePredictionEngine<MushroomModelInput, MushroomModelPrediction>(model);

            var predOutput = predictEngine.Predict(input);

            return predOutput;


        }
             
        
        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
        public static void PeekDataViewInConsole(MLContext mlContext, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
          

            //https://github.com/dotnet/machinelearning/blob/master/docs/code/MlNetCookBook.md#how-do-i-look-at-the-intermediate-data
            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // 'transformedData' is a 'promise' of data, lazy-loading. call Preview  
            //and iterate through the returned collection from preview.

            var preViewTransformedData = transformedData.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                Console.WriteLine(lineToPrint + "\n");
            }
        }

        //     .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
    }
}

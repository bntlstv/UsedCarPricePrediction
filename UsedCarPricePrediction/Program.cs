using Microsoft.ML;
using UsedCarPricePrediction;

var mlContext  = new MLContext();

var data  = mlContext.Data.LoadFromTextFile<UsedCarData>("used_cars_1000.csv", hasHeader: true, separatorChar: ',');

var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

var pipeline = mlContext.Transforms.Text.FeaturizeText(
                                          outputColumnName: "MakeFeatures",
                                          inputColumnName: "Make")
                        .Append(mlContext.Transforms.Text.FeaturizeText(
                                          outputColumnName: "ModelFeatures",
                                          inputColumnName: "Model"))
                        .Append(mlContext.Transforms.Text.FeaturizeText(
                                          outputColumnName: "TransmissionFeatures",
                                          inputColumnName: "Transmission"))
                        .Append(mlContext.Transforms.Text.FeaturizeText(
                                          outputColumnName: "FuelTypeFeatures",
                                          inputColumnName: "FuelType")) 
                        .Append(mlContext.Transforms.Concatenate("Features", "MakeFeatures", "ModelFeatures", "TransmissionFeatures", "FuelTypeFeatures", "Mileage", "EngineSize", "Horsepower", "Doors"))                 
                        .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                        .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", featureColumnName: "Features"));

var model = pipeline.Fit(split.TrainSet);

var predictions = model.Transform(split.TestSet);

var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Price");

Console.WriteLine("Predictions");
Console.WriteLine($"R2:   {metrics.RSquared:0.###}");  
Console.WriteLine($"MAE:  {metrics.MeanAbsoluteError:0.##}");
Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:0.##}");

var predictionEngine = mlContext.Model.CreatePredictionEngine<UsedCarData, UsedCarPrediction>(model);

var sampleData = new UsedCarData
{
  Make = "Toyota",
  Model = "Rav4",
  Year = 2018,
  Mileage = 42000,
  EngineSize = 2.5f,
  Horsepower = 100,
  Transmission = "Automatic",
  FuelType = "Gas",
  Doors = 4
};

Console.WriteLine("");
Console.WriteLine("Prediction Input Data");
Console.WriteLine($"Make         {sampleData.Make}");
Console.WriteLine($"Model        {sampleData.Model}");
Console.WriteLine($"Year         {sampleData.Year}");
Console.WriteLine($"Mileage      {sampleData.Mileage}");
Console.WriteLine($"EngineSize   {sampleData.EngineSize}");
Console.WriteLine($"Horsepower   {sampleData.Horsepower}");
Console.WriteLine($"Transmission {sampleData.Transmission}");
Console.WriteLine($"FuelType     {sampleData.FuelType}");
Console.WriteLine($"Doors        {sampleData.Doors}");

var prediction = predictionEngine.Predict(sampleData);

Console.WriteLine("");
Console.WriteLine($"Predictied Price: ${ prediction.Score:0}");

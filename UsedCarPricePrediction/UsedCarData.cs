using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace UsedCarPricePrediction
{
  public class UsedCarData
  {
    [LoadColumn(0)]
    public required string Make { get; set; }

    [LoadColumn(1)]
    public required string Model { get; set; }

    [LoadColumn(2)]
    public float Year { get; set; }

    [LoadColumn(3)]
    public float Mileage { get; set; }

    [LoadColumn(4)]
    public float EngineSize { get; set; }

    [LoadColumn(5)]
    public float Horsepower { get; set; }

    [LoadColumn(6)]
    public required string Transmission { get; set; }

    [LoadColumn(7)]
    public required string FuelType { get; set; }

    [LoadColumn(8)]
    public float Doors { get; set; }

    [LoadColumn(9)]
    public float Price { get; set; }
  }
}

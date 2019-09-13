using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace MushroomClassifier.DataModels
{
    class MushroomModelPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Label { get; set; }

        public float[] Score { get; set; }
    }
}

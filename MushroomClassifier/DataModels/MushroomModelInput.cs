using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace MushroomClassifier.DataModels
{
    class MushroomModelInput
    {
        [LoadColumn(0)]
        public string mClass { get; set; }

        [LoadColumn(1)]
        public string cap_shape { get; set; }

        [LoadColumn(2)]
        public string cap_surface { get; set; }

        [LoadColumn(3)]
        public string cap_color { get; set; }

        [LoadColumn(4)]
        public string bruises { get; set; }

        [LoadColumn(5)]
        public string odor { get; set; }

        [LoadColumn(6)]
        public string gill_attachment { get; set; }

        [LoadColumn(7)]
        public string gill_spacing { get; set; }

        [LoadColumn(8)]
        public string gill_size { get; set; }

        [LoadColumn(9)]
        public string gill_color { get; set; }

        [LoadColumn(10)]
        public string stalk_shape { get; set; }

        [LoadColumn(11)]
        public string stalk_root { get; set; }

        [LoadColumn(12)]
        public string stalk_surface_above_ring { get; set; }

        [LoadColumn(13)]
        public string stalk_surface_below_ring { get; set; }

        [LoadColumn(14)]
        public string stalk_color_above_ring { get; set; }

        [LoadColumn(15)]
        public string stalk_color_below_ring { get; set; }

        [LoadColumn(16)]
        public string veil_type { get; set; }

        [LoadColumn(17)]
        public string veil_color { get; set; }
        
        [LoadColumn(18)]
        public string ring_number { get; set; }
        
        [LoadColumn(19)]
        public string ring_type { get; set; }

        [LoadColumn(20)]
        public string spore_print_color { get; set; }

        [LoadColumn(21)]
        public string population { get; set; }

        [LoadColumn(22)]
        public string habitat { get; set; }

    }



                          

}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Math
{
    class MathF
    {
        public static double Clamp (double value, double min, double max)
        {
            if (value > max)
                value = max;
            if (value < min)
                value = min;
            return value;
        }
    }
}

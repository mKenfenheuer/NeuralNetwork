using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Math
{
    public class RandomValues
    {
        static Random random = new Random((int)DateTime.Now.Ticks);
        public static double RandomDouble()
        {
            return random.NextDouble();
        }

        public static int RandomInt(int min, int max)
        {
            return random.Next(min, max);
        }
    }
}

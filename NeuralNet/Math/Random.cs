using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Math
{
    public class RandomValues
    {
        private static System.Random random = new System.Random((int)DateTime.Now.Ticks);

        public static double RandomDouble()
        {
            return random.NextDouble();
        }

        public static int RandomInt(int min, int max)
        {
            return (int) (System.Math.Round(random.NextDouble().Map(0, 1, min, max)));
        }
    }
}

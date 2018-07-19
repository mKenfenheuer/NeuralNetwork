using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Threading.Tasks;
using System.Collections;

using NeuralNet.Training.GeneticEvolve;
using NeuralNet.Network.Interfaces;
using NeuralNet.Math;

namespace NeuralNetTester
{
    class Program
    {
        static void Main(string[] args)
        {
            bool test = new NetworkIntegrityTest().Run();
            if (!test)
                Console.WriteLine("Meh!");

            new GeneticEvolveTest().Run().Wait();
            Console.ReadKey();
        }

        
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


using NeuralNet.Network.Implementation;
using NeuralNet.Network.Interfaces;
using NeuralNet.Math;

namespace NeuralNetTester
{
    class NetworkIntegrityTest
    {
        public bool Run()
        {
            NeuralNetwork net = new NeuralNetwork(new int[] { 2, 3, 5, 5, 2, 1 }, Math.Tanh);
            NeuralNetwork net2 = (NeuralNetwork)net.Clone();
            double[] inputs = { 0.734, 0.92764 };
            double net1Out = net.Calculate(inputs)[0];
            double net2Out = net2.Calculate(inputs)[0];
            net2.Mutate(0.3,0.5);
            double net1PostOut = net.Calculate(inputs)[0];
            double net2PostOut = net2.Calculate(inputs)[0];
            bool test1 = net1Out - net1PostOut == 0;
            bool test2 = net2Out - net2PostOut != 0;
            bool test3 = net1Out - net2Out == 0;
            return test1 && test2 && test3;
        }
    }
}

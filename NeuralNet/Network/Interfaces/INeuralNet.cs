using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Network.Interfaces
{
    public interface INeuralNet : ICloneable
    {
        double[] Calculate(double[] inputs);
    }

    internal interface INeuralNetInternal : INeuralNet
    {
        Guid GetGuid();
        double ActivationFunction(double value);
        INeuralNetInternal[] Mutate(double probability, double factor, int networkCount);
    }
}

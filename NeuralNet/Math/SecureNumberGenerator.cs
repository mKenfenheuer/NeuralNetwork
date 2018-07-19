using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Security.Cryptography;

namespace NeuralNet.Math
{
    class SecureNumberGenerator
    {
        RNGCryptoServiceProvider provider = new RNGCryptoServiceProvider();

        public double RandomDouble()
        {
            byte[] data = new byte[sizeof(uint)];
            provider.GetBytes(data);
            uint number = BitConverter.ToUInt32(data, 0);
            double num = number / (double) uint.MaxValue;
            return num;
        }
    }
}

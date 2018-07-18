using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public static class Extensions
    {
        public static List<T> Clone<T>(this List<T> list) where T : ICloneable
        {
            List <T> otherList = new List<T>();
            foreach(T obj in list)
            {
                otherList.Add((T)obj.Clone());
            }
            return otherList;
        }
    }
}

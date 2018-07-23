using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet.Math;
using NeuralNet.Network.Interfaces;
using NeuralNet.Network.Implementation;

namespace NeuralNet.Training.GeneticEvolve
{
    public class GeneticEvolver
    {
        int netsPerPopulation;
        int netPopulations;
        Tuple<int, int> layerRange;
        Tuple<int, int> layerLength;
        int inputs;
        int outputs;
        Func<double, double> activationFunction;
        Func<INeuralNet, double> evaluationFunc;
        Dictionary<Guid, NeuralNetwork> networks = new Dictionary<Guid, NeuralNetwork>();
        Dictionary<Guid, double> fitnesses = new Dictionary<Guid, double>();
        public int Generation => generation;
        int generation;
        public double MaxFitness => maxFitness;
        double maxFitness;
        public INeuralNet BestNetwork => bestNetwork;
        INeuralNet bestNetwork;
        public int InputCount => inputs;
        public int OutputCount => outputs;
        Func<double, double[]> mutationFunction;

        public GeneticEvolver(Tuple<int, int> layerRange,
                                Tuple<int, int> layerLength,
                                Func<INeuralNet, double> evaluationFunc,
                                Func<double, double> activationFunction,
                                Func<double, double[]> mutationFunction,
                                int inputs,
                                int outputs,
                                int netsPerPopulation = 10,
                                int netPopulations = 10)
        {
            this.inputs = inputs;
            this.outputs = outputs;
            this.evaluationFunc = evaluationFunc;
            this.activationFunction = activationFunction;
            this.netsPerPopulation = netsPerPopulation;
            this.netPopulations = netPopulations;
            this.layerRange = layerRange;
            this.layerLength = layerLength;
            this.mutationFunction = mutationFunction;

        }

        public void Init()
        {
            for (int pop = 0; pop < netPopulations; pop++)
            {
                int[] layers = new int[RandomValues.RandomInt(layerLength.Item1, layerLength.Item2)];
                for (int i = 0; i < layers.Length; i++)
                    layers[i] = RandomValues.RandomInt(layerRange.Item1, layerRange.Item2);
                layers[0] = inputs;
                layers[layers.Length - 1] = outputs;
                for (int net = 0; net < netsPerPopulation; net++)
                {
                    NeuralNetwork network = new NeuralNetwork(layers, activationFunction);
                    networks.Add(network.Guid, network);
                }
            }
        }
        public void AddNetwork(NeuralNetwork net)
        {
            if (net.Layers[0] != inputs && net.Layers[net.Layers.Length - 1] != outputs)
                throw new ArgumentException("Network In and Outputs do not match Evolver Parameters");

            networks[net.Guid] = net;
        }
        public async Task Evaluate()
        {
            fitnesses.Clear();
            await Task.WhenAll(networks.Select(kvp => Task.Run(async () =>
            {
                double fitness = 0;
                await Task.Run(() => fitness = evaluationFunc(kvp.Value));
                lock (fitnesses)
                {
                    if (fitnesses.ContainsKey(kvp.Key))
                        fitnesses.Remove(kvp.Key);
                    fitnesses.Add(kvp.Key, fitness);
                }
            }
            )));
            List<NeuralNetwork> networkList = networks.Select(kvp => kvp.Value).ToList();
            networkList.Sort((t1, t2) => { return -fitnesses[t1.GetGuid()].CompareTo(fitnesses[t2.GetGuid()]); });
            maxFitness = fitnesses[networkList[0].GetGuid()];
            bestNetwork = networkList[0];
        }

        public void Evolve()
        {
            List<NeuralNetwork> allNewNetworks = new List<NeuralNetwork>();

            List<NeuralNetwork> networkList = networks.Select(kvp => kvp.Value).ToList();
            networkList.Sort((t1, t2) => { return -fitnesses[t1.GetGuid()].CompareTo(fitnesses[t2.GetGuid()]); });

            List<NeuralNetwork> best10 = networkList.Take((int)(networkList.Count * 0.1)).ToList();

            List<double> fitness = networkList.Select(n => fitnesses[n.GetGuid()]).ToList();

            allNewNetworks.AddRange(networkList.Take(5).Select(n => n));
            double[] mutationParams = mutationFunction(maxFitness);
            allNewNetworks.AddRange(best10.SelectMany(n => n.Mutate(mutationParams[0], mutationParams[1], 4)));

            while (allNewNetworks.Count < networkList.Count * 0.75)
            {
                NeuralNetwork n1 = networkList[RandomValues.RandomInt(0, networkList.Count - 1)];
                NeuralNetwork n2 = networkList[RandomValues.RandomInt(0, networkList.Count - 1)];
                if (n1.WantsSexyTimeWith(n2) && n2.WantsSexyTimeWith(n1))
                    allNewNetworks.Add(n1.DoSexyTimeWith(n2));
            }

            while (allNewNetworks.Count < networkList.Count)
            {
                int[] layers = new int[RandomValues.RandomInt(layerLength.Item1, layerLength.Item2)];
                for (int i = 0; i < layers.Length; i++)
                    layers[i] = RandomValues.RandomInt(layerRange.Item1, layerRange.Item2);
                layers[0] = inputs;
                layers[layers.Length - 1] = outputs;
                NeuralNetwork network = new NeuralNetwork(layers, activationFunction);
                allNewNetworks.Add(network);
            }

            networks.Clear();
            fitnesses.Clear();

            foreach (NeuralNetwork network in allNewNetworks)
                networks.Add(network.GetGuid(), network);

            generation++;
        }

    }
}

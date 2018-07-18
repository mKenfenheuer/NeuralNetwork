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
        double mutationProbability;
        double maxMutationFactor;
        int inputs;
        int outputs;
        Func<double, double> activationFunction;
        Func<INeuralNet, double> evaluationFunc;
        Dictionary<Guid, INeuralNetInternal> networks = new Dictionary<Guid, INeuralNetInternal>();
        Dictionary<Guid, double> fitnesses = new Dictionary<Guid, double>();
        public int Generation => generation;
        int generation;
        public double MaxFitness => maxFitness;
        double maxFitness;
        public INeuralNet BestNetwork => bestNetwork;
        INeuralNet bestNetwork;
        public int InputCount => inputs;
        public int OutputCount => outputs;

        public GeneticEvolver(Tuple<int, int> layerRange,
                                Tuple<int, int> layerLength,
                                Func<INeuralNet, double> evaluationFunc,
                                Func<double, double> activationFunction,
                                int inputs,
                                int outputs,
                                int netsPerPopulation = 10,
                                int netPopulations = 10,
                                double mutationProbability = 0.2,
                                double maxMutationFactor = 0.015)
        {
            this.inputs = inputs;
            this.outputs = outputs;
            this.evaluationFunc = evaluationFunc;
            this.activationFunction = activationFunction;
            this.netsPerPopulation = netsPerPopulation;
            this.netPopulations = netPopulations;
            this.layerRange = layerRange;
            this.layerLength = layerLength;
            this.mutationProbability = mutationProbability;
            this.maxMutationFactor = maxMutationFactor;

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
            List<INeuralNetInternal> networkList = networks.Select(kvp => kvp.Value).ToList();
            networkList.Sort((t1, t2) => { return -fitnesses[t1.GetGuid()].CompareTo(fitnesses[t2.GetGuid()]); });
            maxFitness = fitnesses[networkList[0].GetGuid()];
            bestNetwork = networkList.FirstOrDefault();
        }

        public void Evolve()
        {
            List<INeuralNetInternal> allNewNetworks = new List<INeuralNetInternal>();

            List<INeuralNetInternal> networkList = networks.Select(kvp => kvp.Value).ToList();
            networkList.Sort((t1, t2) => { return -fitnesses[t1.GetGuid()].CompareTo(fitnesses[t2.GetGuid()]); });

            List<INeuralNetInternal> best10 = networkList.Take((int)(networkList.Count * 0.1)).ToList();

            List<double> fitness = networkList.Select(n => fitnesses[n.GetGuid()]).ToList();

            allNewNetworks.AddRange(networkList.Take(5));
            allNewNetworks.AddRange(best10.SelectMany(n => n.Mutate(mutationProbability, maxMutationFactor, 8)));

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

            foreach (INeuralNetInternal network in allNewNetworks)
                networks.Add(network.GetGuid(), network);

            generation++;
        }

    }
}

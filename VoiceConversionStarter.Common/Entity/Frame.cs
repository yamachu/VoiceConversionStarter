using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using NumSharp;

namespace VoiceConversionStarter.Common.Entity
{
    public class Frame
    {
        public float[] Sources { get; set; }
        public float[] Targets { get; set; }

        public static IEnumerable<Frame> FromFile(string sourceFilePath, string targetFilePath)
        {
            if (!File.Exists(sourceFilePath) || !File.Exists(targetFilePath))
                throw new ArgumentException($"{sourceFilePath} or {targetFilePath} are not exist");

            var sourceFeatures = np.Load<float[,]>(sourceFilePath);
            var targetFeatures = np.Load<float[,]>(targetFilePath);

            var featureLength = sourceFeatures.GetLength(0);

            if (featureLength != targetFeatures.GetLength(0))
                throw new RankException($"feature frame must be matched");

            var sourceDim = sourceFeatures.GetLength(1);
            var targetDim = targetFeatures.GetLength(1);

            var s = sourceFeatures.Cast<float>().ToArray();
            var t = targetFeatures.Cast<float>().ToArray();

            foreach (var i in Enumerable.Range(0, featureLength))
                yield return new Frame
                {
                    Sources = new ArraySegment<float>(s, i * sourceDim, sourceDim).ToArray(),
                    Targets = new ArraySegment<float>(t, i * targetDim, targetDim).ToArray()
                };
        }
    }
}

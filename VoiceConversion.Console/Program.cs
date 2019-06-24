using System;
using System.Collections.Immutable;
using System.Globalization;
using System.IO;
using System.Linq;
using CommandLine;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using VoiceConversion.Common.Entity;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace VoiceConversion.Console
{
    class Program
    {
        [Verb("train-mcep", HelpText = "train mcep-model")]

        class TrainMcapOptions
        {
            [Option("base", Required = true, HelpText = "base tensorflow model dir.")]
            public string BaseModel { get; set; }

            [Option("source-dir", Required = true, HelpText = "source features dir.")]
            public string SourceDir { get; set; }

            [Option("target-dir", Required = true, HelpText = "target features dir.")]
            public string TargetDir { get; set; }

            [Option("save-dir", Default = ".", HelpText = "to save model and statistic files dir.")]
            public string SaveDir { get; set; }
        }

        static int Train(TrainMcapOptions opts)
        {
            Console.WriteLine("Run Train");

            var sourceFiles = Directory.GetFiles(opts.SourceDir, "*.npy").OrderBy(n => n);
            var targetFiles = Directory.GetFiles(opts.TargetDir, "*.npy").OrderBy(n => n);

            // assert source and target array length equal
            var datasets = Enumerable.Zip(sourceFiles, targetFiles, (s, t) => Frame.FromFile(s, t)).SelectMany(v => v);

            var template = datasets.First();

            var scheme = SchemaDefinition.Create(typeof(Frame));
            scheme[nameof(Frame.Sources)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, template.Sources.Length);
            scheme[nameof(Frame.Targets)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, template.Targets.Length);

            var mlContext = new MLContext(seed: 555);
            var data = mlContext.Data.LoadFromEnumerable(datasets, scheme);

            var tfInputName = "X";
            var tfOutputName = "Y";

            // set in WithOnFitDelegate, todo: lazy?
            NormalizingTransformer sourceNormalizeTransformer = null;
            NormalizingTransformer targetNormalizeTransformaer = null;

            var preparePipeline = mlContext.Transforms.Concatenate(tfOutputName, nameof(Frame.Targets))
                .Append(mlContext.Transforms.Concatenate(tfInputName, nameof(Frame.Sources)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(tfInputName, useCdf: true)
                    .WithOnFitDelegate(v => sourceNormalizeTransformer = v))
                .Append(mlContext.Transforms.NormalizeMeanVariance(tfOutputName, useCdf: true)
                    .WithOnFitDelegate(v => targetNormalizeTransformaer = v))
                .AppendCacheCheckpoint(mlContext);

            var normalizedData = preparePipeline.Fit(data).Transform(data);

            var pipeline = mlContext.Model.LoadTensorFlowModel(opts.BaseModel)
                        .RetrainTensorFlowModel(
                            inputColumnNames: new[] { tfInputName },
                            outputColumnNames: new[] { "Converted" },
                            labelColumnName: tfOutputName,
                            tensorFlowLabel: tfOutputName,
                            optimizationOperation: "Optimizer",
                            epoch: 1,
                            learningRateOperation: "learning_rate",
                            lossOperation: "Loss"
                        );

            var logger = new ProgressReporter(1);
            mlContext.Log += logger.Log;

            var model = pipeline.Fit(normalizedData);

            // Save model and statistics
            var sep = System.IO.Path.DirectorySeparatorChar;

            using (var stream = File.Create($"{opts.SaveDir}{sep}Model"))
            {
                mlContext.Model.Save(model, data.Schema, stream);
            }

            var sourceMVParams = sourceNormalizeTransformer.GetNormalizerModelParameters(0) as CdfNormalizerModelParameters<ImmutableArray<float>>;
            Common.Util.IO.SaveAsNPY(sourceMVParams.Mean.ToArray(), $"{opts.SaveDir}{sep}Source{sep}Means");
            Common.Util.IO.SaveAsNPY(sourceMVParams.StandardDeviation.ToArray(), $"{opts.SaveDir}{sep}Source{sep}Vars");

            var targetMVParams = targetNormalizeTransformaer.GetNormalizerModelParameters(0) as CdfNormalizerModelParameters<ImmutableArray<float>>;
            Common.Util.IO.SaveAsNPY(targetMVParams.Mean.ToArray(), $"{opts.SaveDir}{sep}Target{sep}Means");
            Common.Util.IO.SaveAsNPY(targetMVParams.StandardDeviation.ToArray(), $"{opts.SaveDir}{sep}Target{sep}Vars");

            return 0;
        }

        static int Main(string[] args)
        {
            return CommandLine.Parser.Default.ParseArguments<TrainMcapOptions>(args)
                .MapResult(
                    (TrainMcapOptions opts) => Train(opts),
                    errs => 1
                );
        }
    }

    class ProgressReporter
    {
        public int FinishCount { get; private set; } = 0;
        public int Epoch { get; }
        private TimeSpan elapsed = TimeSpan.Zero;

        public ProgressReporter(int epoch)
        {
            Epoch = epoch;
        }

        public void Log(object sender, LoggingEventArgs log)
        {
            if (!log.Message.Contains("TensorFlowTransformer")) return;
            if (!log.Message.Contains("Elapsed")) return;
            FinishCount++;
            if (FinishCount > Epoch) return;
            var elapsedLike = log.Message.Split().Last().TrimEnd('.');
            var currentElapsed = TimeSpan.ParseExact(elapsedLike, @"hh\:mm\:ss\.fffffff", CultureInfo.CurrentCulture);
            elapsed += currentElapsed;
            var rest = TimeSpan.FromSeconds((elapsed.TotalSeconds / FinishCount) * Epoch - elapsed.TotalSeconds);
            System.Console.WriteLine($"残り: {rest.ToString(@"hh\:mm\:ss")}, epoch: {FinishCount} / {Epoch}");
        }
    }
}

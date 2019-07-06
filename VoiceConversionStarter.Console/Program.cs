using System;
using System.Collections.Immutable;
using System.Globalization;
using System.IO;
using System.Linq;
using CommandLine;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using VoiceConversionStarter.Common.Entity;
using static Microsoft.ML.Transforms.NormalizingTransformer;

namespace VoiceConversionStarter.Console
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

            [Option("epoch", Default = 20, HelpText = "train epoch.")]
            public int Epoch { get; set; }
        }

        [Verb("eval-mcep", HelpText = "eval mcep-model")]
        class EvalMcapOptions
        {
            [Option("model", Required = true, HelpText = "trained model.")]
            public string Model { get; set; }

            [Option("source-dir", Required = true, HelpText = "source statistics dir.")]
            public string SourceParam { get; set; }

            [Option("target-dir", Required = true, HelpText = "target statistics dir.")]
            public string TargetParam { get; set; }

            [Option('i', "input", Required = true, HelpText = "input npy.")]
            public string Input { get; set; }

            [Option('o', "output", Default = "out.npy", HelpText = "converted npy.")]
            public string Output { get; set; }
        }

        static int Train(TrainMcapOptions opts)
        {
            System.Console.WriteLine("Run Train");
            var sourceFiles = Directory.GetFiles(opts.SourceDir, "*.npy").OrderBy(n => n);
            var targetFiles = Directory.GetFiles(opts.TargetDir, "*.npy").OrderBy(n => n);

            // assert source and target array length equal
            var datasets = Enumerable.Zip(sourceFiles, targetFiles, (s, t) => Frame.FromFile(s, t)).SelectMany(v => v);

            var template = datasets.First();

            var trainScheme = SchemaDefinition.Create(typeof(Frame));
            trainScheme[nameof(Frame.Sources)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, template.Sources.Length);
            trainScheme[nameof(Frame.Targets)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, template.Targets.Length);

            var mlContext = new MLContext(seed: 555);
            var data = mlContext.Data.LoadFromEnumerable(datasets, trainScheme);

            var tfInputName = "X";
            var tfOutputName = "Y";

            // set in WithOnFitDelegate, todo: lazy?
            NormalizingTransformer sourceNormalizeTransformer = null;
            NormalizingTransformer targetNormalizeTransformaer = null;

            var preparePipeline = mlContext.Transforms.Concatenate($"{tfOutputName}_Raw", nameof(Frame.Targets))
                .Append(mlContext.Transforms.Concatenate($"{tfInputName}_Raw", nameof(Frame.Sources)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(tfInputName, $"{tfInputName}_Raw", fixZero: false)
                    .WithOnFitDelegate(v => sourceNormalizeTransformer = v))
                .Append(mlContext.Transforms.NormalizeMeanVariance(tfOutputName, $"{tfOutputName}_Raw", fixZero: false)
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
                            epoch: opts.Epoch,
                            learningRateOperation: "learning_rate",
                            lossOperation: "Loss"
                        );

            var logger = new ProgressReporter(opts.Epoch);
            mlContext.Log += logger.Log;

            var model = pipeline.Fit(normalizedData);

            // Save model and statistics
            var sep = System.IO.Path.DirectorySeparatorChar;

            Common.Util.IO.CreateDirectory(opts.SaveDir);
            using (var stream = File.Create($"{opts.SaveDir}{sep}Model"))
            {
                var tfInputScheme = SchemaDefinition.Create(typeof(SourceFrame));
                tfInputScheme[nameof(SourceFrame.X)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, template.Sources.Length);

                var tfInputDataScheme = mlContext.Data.LoadFromEnumerable(new SourceFrame[] { }, tfInputScheme);

                mlContext.Model.Save(model, tfInputDataScheme.Schema, stream);
            }

            var sourceMVParams = sourceNormalizeTransformer.GetNormalizerModelParameters(0) as AffineNormalizerModelParameters<ImmutableArray<float>>;
            Common.Util.IO.SaveAsNPY(sourceMVParams.Offset.ToArray(), $"{opts.SaveDir}{sep}Source{sep}Means");
            Common.Util.IO.SaveAsNPY(sourceMVParams.Scale.ToArray(), $"{opts.SaveDir}{sep}Source{sep}Vars");

            var targetMVParams = targetNormalizeTransformaer.GetNormalizerModelParameters(0) as AffineNormalizerModelParameters<ImmutableArray<float>>;
            Common.Util.IO.SaveAsNPY(targetMVParams.Offset.ToArray(), $"{opts.SaveDir}{sep}Target{sep}Means");
            Common.Util.IO.SaveAsNPY(targetMVParams.Scale.ToArray(), $"{opts.SaveDir}{sep}Target{sep}Vars");

            return 0;
        }

        static int Eval(EvalMcapOptions opts)
        {
            var mlContext = new MLContext(seed: 555);

            var sep = System.IO.Path.DirectorySeparatorChar;

            var sourceMean = Common.Util.IO.LoadNPY<float[]>($"{opts.SourceParam}{sep}Means.npy");
            var sourceVar = Common.Util.IO.LoadNPY<float[]>($"{opts.SourceParam}{sep}Vars.npy");

            var targetMean = Common.Util.IO.LoadNPY<float[]>($"{opts.TargetParam}{sep}Means.npy");
            var targetVar = Common.Util.IO.LoadNPY<float[]>($"{opts.TargetParam}{sep}Vars.npy");

            var tfModel = mlContext.Model.Load(opts.Model, out var inputSchema);

            var schema = SchemaDefinition.Create(typeof(SourceFrame));
            var schemaInputCol = inputSchema.Single((v) => v.Name == nameof(SourceFrame.X));
            schema[schemaInputCol.Name].ColumnType = schemaInputCol.Type;

            var predictor = mlContext.Model.CreatePredictionEngine<SourceFrame, TargetFrame>(tfModel, inputSchemaDefinition: schema);

            var data = Common.Util.IO.LoadNPY<float[,]>(opts.Input);
            var sourceFrames = Common.Util.Matrix.To2JaggedArray(data).Select(v => new SourceFrame { X = v });

            var converted = sourceFrames
            .Select(v => Common.Util.Matrix.Mul(Common.Util.Matrix.Sub(v.X, sourceMean), sourceVar))
            .Select(v => predictor.Predict(new SourceFrame { X = v }))
            .Select(v => Common.Util.Matrix.Div(Common.Util.Matrix.Add(v.Converted, targetMean), targetVar));

            Common.Util.IO.SaveAsNPY(Common.Util.Matrix.To2DimArray(converted.ToArray()), opts.Output);

            return 0;
        }

        static int Main(string[] args)
        {
            return CommandLine.Parser.Default.ParseArguments<TrainMcapOptions, EvalMcapOptions>(args)
                .MapResult(
                    (TrainMcapOptions opts) => Train(opts),
                    (EvalMcapOptions opts) => Eval(opts),
                    errs => 1
                );
        }
    }

    class ProgressReporter
    {
        public int FinishCount { get; private set; } = 0;
        public int Epoch { get; }
        public TimeSpan Elapsed { get; private set; } = TimeSpan.Zero;

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
            Elapsed += currentElapsed;
            var rest = TimeSpan.FromSeconds((Elapsed.TotalSeconds / FinishCount) * Epoch - Elapsed.TotalSeconds);
            System.Console.WriteLine($"残り: {rest.ToString(@"hh\:mm\:ss")}, epoch: {FinishCount} / {Epoch}");
        }
    }
}

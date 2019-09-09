using System.Linq;
using NumSharp;

namespace VoiceConversionStarter.Common.Util
{
    public static class Matrix
    {
        public static T[,] To2DimArray<T>(T[][] arr)
        {
            return np.array(arr).ToMuliDimArray<T>() as T[,];
        }

        public static T[][] To2JaggedArray<T>(T[,] arr)
        {
            return np.array(arr).ToJaggedArray<T>() as T[][];
        }

        public static T[] Add<T>(T[] first, T[] second)
        {
            return (np.array(first) + np.array(second)).Cast<T>().ToArray();
        }

        public static T[] Sub<T>(T[] first, T[] second)
        {
            return (np.array(first) - np.array(second)).Cast<T>().ToArray();
        }

        public static T[] Mul<T>(T[] first, T[] second)
        {
            return (np.array(first) * np.array(second)).Cast<T>().ToArray();
        }

        public static T[] Div<T>(T[] first, T[] second)
        {
            return (np.array(first) / np.array(second)).Cast<T>().ToArray();
        }
    }
}

using System.Linq;
using NumSharp;

namespace VoiceConversionStarter.Common.Util
{
    public static class IO
    {
        public static void SaveAsNPY<T>(T[] arr, string path)
        {
            var dirSeparate = path.Split(new[] { System.IO.Path.DirectorySeparatorChar });
            var dir = string.Join(System.IO.Path.DirectorySeparatorChar.ToString(), dirSeparate.Take(dirSeparate.Length - 1));
            System.IO.Directory.CreateDirectory(dir);
            np.Save(arr, path);
        }
    }
}

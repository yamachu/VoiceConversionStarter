using System;
using System.Collections;
using System.Linq;
using NumSharp;

namespace VoiceConversionStarter.Common.Util
{
    public static class IO
    {
        public static void SaveAsNPY(Array arr, string path)
        {
            var dirSeparate = path.Split(new[] { System.IO.Path.DirectorySeparatorChar });
            var dir = string.Join(System.IO.Path.DirectorySeparatorChar.ToString(), dirSeparate.Take(dirSeparate.Length - 1));
            if (dir != "") CreateDirectory(dir);
            np.Save(arr, path);
        }

        public static T LoadNPY<T>(string path) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return np.Load<T>(path: path);
        }

        public static void CreateDirectory(string path)
        {
            if (System.IO.Directory.Exists(path)) return;
            System.IO.Directory.CreateDirectory(path);
        }
    }
}

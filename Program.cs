/*
 ----------Diyari ismaeil----------
 ----Iraqi dinar classification----
 ------------15\12\2020------------
*/
using Keras;
using Keras.Models;
using Keras.PreProcessing.Image;
using Numpy;
using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace IQD_Classification
{
    class Program
    {
        private static string[] labels = new string[] { "1000 IQD", "10000 IQD", " 250 IQD", "25000 IQD", "500 IQD", "5000 IQD", "50000 IQD" };

        static void Main(string[] args)
        {

            Console.Write("Path : ");
            string Path = Console.ReadLine();
            Console.WriteLine("Running");

            string result = Classification(Path);
            Console.WriteLine(result);
            #region
            Form imageFrm = new Form();
            PictureBox pictureBox = new PictureBox();
            pictureBox.Image = Image.FromFile(Path);
            Label label = new Label();
            imageFrm.Size = new Size(400, 400);
            imageFrm.Controls.Add(pictureBox);
            imageFrm.Controls.Add(label);
            label.Location = new Point(0, 20);

            pictureBox.Dock = DockStyle.Fill;
            pictureBox.SizeMode = PictureBoxSizeMode.Zoom;

            label.BringToFront();
            label.BackColor = Color.Transparent;
            label.Anchor = (AnchorStyles.Top | AnchorStyles.Left);
            label.Font = new Font(FontFamily.GenericSansSerif, 12, FontStyle.Bold);
            label.Text = result;
            imageFrm.ShowDialog();
            #endregion

            Console.ReadKey();
        }
        public static string Classification(string path)
        {
            string rv = "";

            string imagePath = Path.GetFullPath(path);
            string modelPath = Path.GetFullPath("Diyari_IQD.h5");
            string weightsPath = Path.GetFullPath("Diyari_IQD_weights.h5");

            if (File.Exists(imagePath))
            {
                var img = ImageUtil.LoadImg(path, target_size: new Shape(64, 64));
                NDarray x = ImageUtil.ImageToArray(img);
                x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2]);
                var model = Sequential.LoadModel(modelPath);
                model.LoadWeight(weightsPath);
                var y = model.Predict(x);
                y = y.argmax();
                int index = y.asscalar<int>();
                rv = labels[index];
            }
            else
            {
                throw (new Exception("No Image found at: " + imagePath));
            }

            return rv;
        }
    }
}

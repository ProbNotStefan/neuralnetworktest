#include <bits/stdc++.h>
#include <fstream>
using namespace std;

string fileloc = "test.nn";

double dotmul(vector<double> v1, vector<double> v2)
{
    double res = 0.0;
    for (int i = 0; i < v1.size(); ++i)
    {
        res += v1[i] * v2[i];
    }
    return res;
}



string planeflatten(vector<vector<double>> data)
{
    string flat = "";
    for (int rowi = 0; rowi < data.size(); ++rowi)
    {
        for (int coli = 0; coli < data[0].size(); ++coli)
        {
            flat += to_string(data[rowi][coli]) + " ";
        }
    }
    return flat.substr(0,flat.size()-1);
}

string flatten(vector<int> data)
{
    string flat = "";
    for (int i = 0; i < data.size(); ++i)
    {
        flat += to_string(data[i]) + " ";
    }
    return flat.substr(0,flat.size()-1);
}

struct matrix
{
    public:
        vector<vector<double>> data;
        int cols;
        int rows;
        void rinit()
        {
            data.resize(rows, vector<double>(cols)); 
            uniform_real_distribution<double> unif(-2.0,2.0);
            default_random_engine re(time(0));
            for (int rowi = 0; rowi < rows; ++rowi)
            {
                for (int coli = 0; coli < cols; ++coli)
                {
                    data[rowi][coli] = unif(re);
                }
            }
        }
        void zinit()
        {
            data.resize(rows, vector<double>(cols, 0.0));
        }
        void output()
        {
            for (int rowi = 0; rowi < rows; ++rowi)
            {
                for (int coli = 0; coli < cols; ++coli)
                {
                    cout << data[rowi][coli] << " ";
                }
                cout << endl;
            }
        }
        void activation()
        {
            for (int rowi = 0; rowi < rows; ++rowi)
            {
                for (int coli = 0; coli < cols; ++coli)
                {
                    //sigmoid
                    data[rowi][coli] = 1.0/(1.0+exp(-data[rowi][coli]));
                }
            }
        }
        matrix dactivation()
        {
            matrix da;
            da.rows = rows;
            da.cols = cols;
            da.zinit();
            for (int rowi = 0; rowi < rows; ++rowi)
            {
                for (int coli = 0; coli < cols; ++coli)
                {
                    //dif sigmoid
                    da.data[rowi][coli] = data[rowi][coli]*(1.0-data[rowi][coli]);
                }
            }
            return da;
        }
        void transpose()
        {
            vector<vector<double>> tdata(cols, vector<double>(rows));
            for (int rowi = 0; rowi < rows; rowi++)
            {
                for (int coli = 0; coli < cols; coli++)
                {
                    tdata[coli][rowi] = data[rowi][coli];
                }
            }
            data = tdata;
            swap(rows, cols);
        }
        matrix ttranspose()
            {
                matrix t;
                t.rows = rows;
                t.cols = cols;
                t.data = data;
                t.transpose();
                return t;
            }
        vector<double> getrow(int rowi)
        {
            return data[rowi];
        }
        vector<double> getcol(int coli)
        {
            vector<double> col; 
            for (int rowi = 0; rowi < rows; ++rowi)
            {
                col.push_back(data[rowi][coli]); 
            }
            return col;
        }
        void replace(int row, int col, double x)
        {
            data[row][col] = x;
        }
        matrix operator+ (matrix m) 
        {
            matrix res;
            res.rows = rows;
            res.cols = cols;
            res.data = data;
            for (int rowi = 0; rowi < rows; ++rowi) 
            {
                for (int coli = 0; coli < cols; ++coli) 
                {
                    res.data[rowi][coli] += m.data[rowi][coli]; 
                }
            }
            return res;
        }
        void operator+= (matrix m) 
        {
            for (int rowi = 0; rowi < rows; ++rowi) 
            {
                for (int coli = 0; coli < cols; ++coli) 
                {
                    data[rowi][coli] += m.data[rowi][coli]; 
                }
            }
        }
        void sub(matrix m) 
        {
            for (int rowi = 0; rowi < rows; ++rowi) 
            {
                for (int coli = 0; coli < cols; ++coli) 
                {
                    data[rowi][coli] -= m.data[rowi][coli]; 
                }
            }
        }
        void mulall(double x) 
        {
            for (int rowi = 0; rowi < rows; ++rowi) 
            {
                for (int coli = 0; coli < cols; ++coli) 
                {
                    data[rowi][coli] *= x; 
                }
            }
        }
        matrix mmulall(matrix m) 
        {
            matrix mm;
            mm.rows = rows;
            mm.cols = cols;
            mm.zinit();
            for (int rowi = 0; rowi < rows; ++rowi) 
            {
                for (int coli = 0; coli < cols; ++coli) 
                {
                   mm.data[rowi][coli] = data[rowi][coli]*m.data[rowi][coli]; 
                }
            }
            return mm;
        }
        matrix sum()
        {
            matrix s;
            s.rows = rows;
            s.cols = 1;
            s.zinit();
            for (int rowi = 0; rowi < rows; ++rowi) 
            {
                for (int coli = 0; coli < cols; ++coli) 
                {
                    s.data[rowi][0] += data[rowi][coli]; 
                }
            }
            return s;
        }
        matrix operator*(matrix & m)
        {
            if (cols != m.rows)
            {
                cerr << "Matrix multiplication error: Mismatched sizes" << endl;
            }
            matrix res;
            res.cols = m.cols;
            res.rows = rows;
            res.zinit();
            for (int rowi = 0; rowi < rows; ++rowi)
            {
                for (int coli = 0; coli < m.cols; ++coli)
                {
                    res.replace(rowi,coli,dotmul(getrow(rowi),m.getcol(coli)));
                }   
            }
            return res;
        }
};

void compsize(matrix a, matrix b)
{
    cout << a.rows << "x" << a.cols << " " << b.rows << "x" << b.cols << endl;
}

matrix matrixify(vector<vector<double>> vec)
{
    matrix m;
    m.rows = vec.size();
    m.cols = vec[0].size();
    m.data = vec;
    return m;
}

vector<matrix> massadd(vector<matrix> a, vector<matrix> b)
{
    vector<matrix> t = a;
    for (int i = 0; i < a.size(); ++i)
    {
        t[i] += b[i];
    }
    return t;
}


double cost(matrix yhat,matrix y)
{
    double sum = 0.0;
    for (int i = 0; i < y.rows; ++i)
    {
        sum += - (y.data[i][0] * log( yhat.data[i][0] )  +  (1 - y.data[i][0]) * log( 1 - yhat.data[i][0] ));
    }
    return sum / (1.0 * y.rows);
}

matrix matrixmul(matrix m1, matrix m2)
{
    if (m1.cols != m2.rows)
    {
        cerr << "Matrix multiplication error: Mismatched sizes" << endl;
    }
    matrix res;
    res.cols = m2.cols;
    res.rows = m1.rows;
    res.zinit();
    for (int rowi = 0; rowi < m1.rows; ++rowi)
    {
        for (int coli = 0; coli < m2.cols; ++coli)
        {
            res.replace(rowi,coli,dotmul(m1.getrow(rowi),m2.getcol(coli)));
        }   
    }
    return res;
}

struct neuralnetwork : public matrix 
{
    public:
        int numlayers;
        vector<int> numnodes;   
        vector<matrix> inputdata;
        vector<matrix> labels;
        vector<matrix> nodes;
        vector<matrix> weights;
        vector<matrix> biases;
        
        void init()
        {
            nodes.resize(numlayers);
            weights.resize(numlayers - 1);
            biases.resize(numlayers - 1);
            nodes[0].rows = numnodes[0];
            nodes[0].cols = 1;
            nodes[0].zinit();
            for (int i = 0; i < numlayers - 1; ++i)
            {
                biases[i].rows = numnodes[i+1];
                biases[i].cols = 1;
                biases[i].rinit();
                weights[i].rows = numnodes[i+1];
                weights[i].cols = numnodes[i];
                weights[i].rinit();
                nodes[i + 1].rows = numnodes[i+1];
                nodes[i + 1].cols = 1;
                nodes[i + 1].zinit();
            }
        }
        void resetnodes()
        {
            nodes[0].rows = numnodes[0];
            nodes[0].cols = 1;
            nodes[0].zinit();
            for (int i = 0; i < numlayers - 1; ++i)
            {
                nodes[i + 1].rows = numnodes[i+1];
                nodes[i + 1].cols = 1;
                nodes[i + 1].zinit();
            }
        }
        void save(string fileloc) {
            ofstream fout(fileloc, ios::out); 
            if (fout.is_open()) 
            {
                fout << numlayers << endl;
                fout << flatten(numnodes) << endl; 
                for (int i = 0; i < numlayers - 1; ++i) 
                {
                    fout << planeflatten(weights[i].data) << endl; 
                }
                for (int i = 0; i < numlayers - 1; ++i) 
                {
                    fout << planeflatten(biases[i].data) << endl; 
                }
                fout.close();
                cout << "File saved successfully." << endl;
            } 
            else 
            {
                cerr << "Save Error: Could not open file " << fileloc << endl;
            }
        }
        void load(string fileloc) {
            ifstream fin(fileloc, ios::in); 
            if (fin.is_open()) 
            {
                fin >> numlayers;
                numnodes.resize(numlayers);
                for (int i = 0; i < numlayers; ++i)
                {
                    fin >> numnodes[i];
                }
                weights.resize(numlayers-1);
                biases.resize(numlayers-1);
                for (int i = 0; i < numlayers - 1; ++i) 
                {
                    vector<vector<double>> tweight = {};
                    tweight.resize(numnodes[i+1], vector<double>(numnodes[i]));
                    for (int rowi = 0; rowi < numnodes[i+1]; ++rowi)
                    {
                        for (int coli = 0; coli < numnodes[i]; ++coli)
                        {
                            fin >> tweight[rowi][coli];
                        }
                    }
                    weights[i] = matrixify(tweight);
                }
                
                for (int i = 0; i < numlayers - 1; ++i) 
                {
                    vector<double> tbias = {};
                    tbias.resize(numnodes[i+1]);
                    for (int j = 0; j < numnodes[i+1]; ++j)
                    {
                        fin >> tbias[j];
                    }
                    biases[i] = matrixify({tbias});
                    biases[i].transpose();
                }
                fin.close();
                nodes.resize(numlayers);
                resetnodes();
                cout << "File loaded successfully." << endl;
            } 
            else 
            {
                cerr << "Load Error: Could not open file " << fileloc << endl;
            }
        }
        matrix feedforward(int i)
        {   
            resetnodes();
            nodes[0] = inputdata[i];
            for (int layer = 0; layer < numlayers - 1; ++layer)
            {
                nodes[layer+1] += weights[layer] * nodes[layer] + biases[layer];
                nodes[layer+1].activation();
            }
            return nodes.back();
        }
        vector<vector<matrix>> backpropagation(matrix yhat, matrix y, int m)
        {
            static vector<matrix> dCdW(numlayers - 1);
            static vector<matrix> dCdb(numlayers - 1);
            static vector<matrix> dCdZ(numlayers - 1);
            static vector<matrix> dCdA(numlayers - 2);
            // backpropagate last layer
            dCdZ[0] = yhat;
            dCdZ[0].sub(y);
            dCdZ[0].mulall(1.0/m);
            matrix tm = y.ttranspose();
            dCdW[0] = (dCdZ[0]) * (tm);
            dCdb[0] = dCdZ[0].sum();
            dCdA[0] = weights[numlayers-2].ttranspose() * dCdZ[0];
            // backpropagate other layers
            for (int i = 1; i < numlayers - 2; ++i)
            {
                dCdZ[i] = dCdA[i-1].mmulall(nodes[numlayers - 1 - i].dactivation());
                tm = nodes[numlayers - 2 - i].ttranspose();
                dCdW[i] = dCdZ[i-1] * tm;
                dCdb[i] = dCdW[i].sum();
                dCdA[i] = weights[numlayers - 2 - i].ttranspose() * dCdZ[i];
            }
            // backpropagate first layer
            dCdZ[numlayers-2] = dCdA[numlayers - 2].mmulall(nodes[1].dactivation());
            tm = nodes[0].ttranspose();
            dCdW[numlayers-2] = dCdZ[numlayers - 3] * tm;
            dCdb[numlayers-2] = dCdW[numlayers - 2].sum();
            return {dCdW,dCdb};
        }

        vector<double> train(int epochs, double alpha)
        {
            vector<double> costs;
            for (int e = 0; e < epochs; ++e)
            {
                double acost = 0.0;
                vector<matrix> dCdW ;
                vector<matrix> dCdb;
                for (int mi = 0; mi < inputdata.size(); ++mi)
                {
                    cout << mi << endl;
                    feedforward(mi);
                    acost += cost(nodes.back(),labels[mi])/inputdata.size();
                    vector<vector<matrix>> dCs = backpropagation(nodes.back(),labels[mi],inputdata.size());
                    if (mi == 0) 
                    {
                        dCdW = dCs[0];
                        dCdb = dCs[1];
                    }
                    else
                    {
                        dCdW = massadd(dCdW, dCs[0]);
                        dCdb = massadd(dCdb, dCs[1]);
                    }
                }
                for (int l = 0; l < numlayers - 1; ++l)
                {   
                    dCdW[l].mulall(alpha);
                    weights[numlayers-1-l].sub(dCdW[l]);
                    dCdb[l].mulall(alpha);
                    biases[numlayers-1-l].sub(dCdb[l]);
                }
                save("C:\\Users\\djuma\\OneDrive\\Documents\\CS\\DEV\\C++\\AI\\test.nn");
                if (epochs > 100)
                {
                    if ((e+1) % 100 == 0)
                    {
                        cout << "epoch " << e+1 << " - cost: " << acost << endl;
                        //costs.push_back(acost);
                    }
                }
                else
                {
                    cout << "epoch " << e+1 << "- cost: " << acost << endl;
                    //costs.push_back(acost);
                }
            }
            return costs;
        }
};

vector<vector<unsigned char>> load_mnist_images(const string& filename, int num_images, int image_width, int image_height) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not open image file: " << filename << endl;
        exit(1);
    }

    vector<vector<unsigned char>> images;
    images.reserve(num_images);

    file.seekg(16, ios::beg); 
    for (int i = 0; i < num_images; ++i) {
        vector<unsigned char> image(image_width * image_height);
        file.read(reinterpret_cast<char*>(image.data()), image_width * image_height);
        images.push_back(image);
    }

    file.close();
    return images;
}

vector<double> load_mnist_labels(const string& filename, int num_labels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not open label file: " << filename << endl;
        exit(1);
    }

    vector<double> labels;
    labels.reserve(num_labels);

    file.seekg(8, ios::beg); 
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(label*1.0);
    }

    file.close();
    return labels;
}

vector<matrix> getmnist() {
    const int num_images = 60000; 
    const int image_width = 28;
    const int image_height = 28;
    vector<matrix> inpu;

    vector<vector<unsigned char>> images = load_mnist_images("C:\\Users\\djuma\\OneDrive\\Documents\\CS\\DEV\\C++\\AI\\train-images.idx3-ubyte", num_images, image_width, image_height);
    images.resize(num_images);
    int i = 0;
    for (const auto& image : images) {
        ++i;
        if ((i)%100 == 0)
        {
            cout << i << "/60000" << endl;
        }
        vector<double> image_1d(image.begin(), image.end()); 
        for (auto& pixel : image_1d) {
            pixel = static_cast<double>(pixel) / 255.0f; 
        }
        inpu.push_back(matrixify({image_1d}).ttranspose());
    }
    return inpu;
}

int main()
{   
    bool read;
    bool dotrain;
    cout << "read?: ";
    cin >> read;
    string fileloc = "C:\\Users\\djuma\\OneDrive\\Documents\\CS\\DEV\\C++\\AI\\test.nn";
    neuralnetwork nn;

    if (read == 1)
    {
        nn.load(fileloc);
    }
    else
    {
        nn.numlayers = 4;
        nn.numnodes = {784, 16, 16, 10};
        nn.init();
        nn.save(fileloc);
    }
    cout << "train?: ";
    cin >> dotrain;
    if (dotrain == 1)
    {
        vector<matrix> input = getmnist();
        cout << "Mnist data loaded." << endl;
        vector<double> labels = load_mnist_labels("C:\\Users\\djuma\\OneDrive\\Documents\\CS\\DEV\\C++\\AI\\train-images.idx3-ubyte", 60000);
        cout << "Mnist labels loaded.\n Transfering to nerual network." << endl;
        for (int inputi = 0; inputi < 60000; ++inputi)
        {
            nn.inputdata.push_back(input[inputi]);
            vector<double> lb = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
            matrix x;
            x.rows = 1;
            x.cols = 10;
            x.data = {lb};
            nn.labels.push_back(x.ttranspose());
        }
        cout << "Started Training" << endl;
        nn.train(3,0.1);
        cout << "Training finished." << endl;
        nn.save(fileloc);
        }
        else
        {
            vector<double> inp(nn.numnodes[0]);
            for (int i = 0; i < nn.numnodes[0]; ++i)
            {
                cin >> inp[i];
            }
            nn.inputdata.push_back(matrixify({inp}).ttranspose());
            nn.feedforward(0);
            nn.nodes.back().output();
        }
    return 0;
}

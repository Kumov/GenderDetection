#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

//face detection

Mat face_detection(Mat img)
{
    CascadeClassifier face_cascade;
    
    face_cascade.load("/Applications/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");
         
    vector<Rect> faces;
    
    face_cascade.detectMultiScale( img, faces, 1.11,4,0,Size(40, 40));
    
    Rect face_pt;
    
    for( int i = 0; i < faces.size(); i++ )
    {

        face_pt.x = faces[i].x;
        face_pt.y = faces[i].y;
        face_pt.width = (faces[i].width);
        face_pt.height = (faces[i].height);
        
        Point pt1(faces[i].x, faces[i].y); // Display detected faces on main window - live stream from camera
        Point pt2((faces[i].x + faces[i].height), (faces[i].y + faces[i].width));
        
        rectangle(img, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    }
    

    Mat img2=img(face_pt);
    return img2;
}

//tan_triggs

Mat tan_triggs_preprocessing(InputArray src,
        float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
        int sigma1 = 2) {

    // Convert to floating point:
    Mat X = src.getMat();
    X.convertTo(X, CV_32FC1);
    // Start preprocessing:
    Mat I;
    pow(X, gamma, I);
    // Calculate the DOG Image:
    {
        Mat gaussian0, gaussian1;
        // Kernel Size:
        int kernel_sz0 = (3*sigma0);
        int kernel_sz1 = (3*sigma1);
        // Make them odd for OpenCV:
        kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
        kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
        GaussianBlur(I, gaussian0, Size(kernel_sz0,kernel_sz0), sigma0, sigma0, BORDER_REPLICATE);
        GaussianBlur(I, gaussian1, Size(kernel_sz1,kernel_sz1), sigma1, sigma1, BORDER_REPLICATE);
        subtract(gaussian0, gaussian1, I);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(abs(I), alpha, tmp);
            meanI = mean(tmp).val[0];

        }
        I = I / pow(meanI, 1.0/alpha);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(min(abs(I), tau), alpha, tmp);
            meanI = mean(tmp).val[0];
        }
        I = I / pow(meanI, 1.0/alpha);
    }

    // Squash into the tanh:
    {
        Mat exp_x, exp_negx;
	exp( I / tau, exp_x );
	exp( -I / tau, exp_negx );
	divide( exp_x - exp_negx, exp_x + exp_negx, I );
        I = tau * I;
    }
    return I;
}

//main function

int main(int, char**)
{
   cout<< "start processing"<<endl;
   
//defining image path and number of train image
   
   char male_image[100]="/Users/HubinoMac2/Documents/NetBeansProjects/sssvm_tantrig/male/male";
   char female_image[100]="/Users/HubinoMac2/Documents/NetBeansProjects/sssvm_tantrig/female/female";
   int nummales=200;
   int numfemales=200;
 
//declaration of matrix
   
   Mat classes;
   Mat trainingData;
   Mat trainingImages;
   Mat img;
   Mat imgs;
   Mat full_image;
   vector<int> trainingLabels;
      
//assigning label to the male images 
  
   for(int i=1; i<= nummales; i++)
    {

        stringstream ss(stringstream::in | stringstream::out);
        ss << male_image << i << ".jpg";
        full_image=imread(ss.str(), 0);  //loading images
        imgs = face_detection(full_image);
        img = tan_triggs_preprocessing(imgs);
        resize(img,img,Size(200,200));//resizing all the images into same size
        img= img.reshape(0, 1);
        trainingImages.push_back(img);
        trainingLabels.push_back(1);
    }
   
   cout<<"male images trained"<<endl;
    
//assigning label to the female images
   
   for(int i=1; i<= numfemales; i++)
    {
        stringstream ss(stringstream::in | stringstream::out);
        ss << female_image << i << ".jpg";
        full_image=imread(ss.str(), 0); //loading images
        imgs = face_detection(full_image);
        img = tan_triggs_preprocessing(imgs);
        resize(img,img,Size(200,200));//resizing all the images into same size
        img= img.reshape(0, 1);
        trainingImages.push_back(img);
        trainingLabels.push_back(0);

    }
   
   cout<<"female images trained"<<endl;
   
//converting image values into float value
   
   Mat tra_set(400,40000,CV_32FC1);
   Mat(trainingImages).copyTo(trainingData);
   trainingData = trainingData.reshape(0,400);
   trainingData.convertTo(tra_set, CV_32FC1,1.0/255.0);
   Mat(trainingLabels).copyTo(classes);
   
//creating XML value
   
    FileStorage fs("sharma.xml", FileStorage::WRITE);
    fs << "TrainingData" << tra_set;
    fs << "classes" << classes;
    fs.release();
    cout<<"training done"<<endl;
            
// Training the SVM
    
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    //svm->setNu(0.09);
    //svm->setC(1);
    //svm->setGamma('auto');
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(tra_set, ROW_SAMPLE, classes);
    svm->save("gender.xml");
    Ptr<SVM> svm1 = StatModel::load<SVM>("gender.xml");
    
// recognizing 
    
    cout<<"recognising...."<<endl;
    
//loading test image

    Mat testing=imread("male20.jpg",0); 
    imshow("original",testing);
    waitKey(0);
    Mat test_face = face_detection(testing);
    imshow("face",test_face);
    waitKey(0);
    Mat testimage = tan_triggs_preprocessing( test_face);
    imshow("face pre proccessed",testimage);
    waitKey(0);
    resize(testimage,testimage,Size(200,200));// resizing into the size of training set
   imshow("resized",testimage);
    waitKey(0);
    testimage = testimage.reshape(0,1);
    
//reshaping and converting into float
    
    Mat test=Mat(testimage.size(),CV_32FC1);
    testimage.convertTo(test, CV_32FC1,1.0/255.0);
    
//predicting SVM
    
    float res= svm->predict(test);
    
//assigning response
    
    cout<<res<<endl;
    res==1? cout<<"male":cout<<"female";
    
}
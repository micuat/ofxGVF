///////////////////////////////////////////////////////////////////////
//
//  GVF - Gesture Variation Follower Max/MSP Object
//
//
//  Copyright (C) 2013 Baptiste Caramiaux, Goldsmiths College, University of London
//
//  The GVF library is under the GNU Lesser General Public License (LGPL v3)
//  version: 09-2013
//
//
///////////////////////////////////////////////////////////////////////


#include "maxcpp6.h"
#include "GVF.h"
#include <string>
#include <map>
#include <vector>
#include <unistd.h>

using namespace std;


enum {STATE_CLEAR, STATE_LEARNING, STATE_FOLLOWING};

int restarted_l;
int restarted_d;

vector<float> vect_0_l;
vector<float> vect_0_d;




class gvf : public MaxCpp6<gvf> {
    
private:
    
    
    // Defining object variables
    GVF             *bubi;
    int             state;
    int             lastreferencelearned;
    int             currentToBeLearned;
    float           sp, sv, sr, ss, smoothingCoef; // pos,vel,rot,scal,observation
	int             pdim;
    vector<float>   means;
    vector<float>   ranges;
    float           value_mmax;
    bool            offline_recognition;
    int             toBeTranslated;
    int             inputDim;
    int             ns;
    int             rt;
    std::map<int,std::vector<std::pair<float,float> > > *refmap;
    
    GVF::GVFVarianceCoefficents coefficients;
    GVF::GVFParameters parameters;
    
public:
    
    
    gvf(t_symbol * sym, long argc, t_atom *argv)
    {
        
        setupIO(1, 3); // inlets / outlets
        post("GVF - realtime adaptive gesture recognition (version: 02-2014)");
        post("(c) Goldsmiths, University of London and Ircam - Centre Pompidou");
        
        
        // PARAMETERS
        
        inputDim = 3;   // input dimensions (default: 2-d gestures)
        ns = 2000;      // number of particles
        rt = 500;       // resampling threshold
        smoothingCoef = 1.7; // smoothing coef (or tolerance)
        
        parameters.inputDimensions = inputDim;
        parameters.numberParticles = ns;
        parameters.tolerance = smoothingCoef;
        parameters.resamplingThreshold = rt;
        parameters.distribution = 0.0f;
        parameters.translate = true;
        parameters.allowSegmentation = false;
        
        
        // COEFFICIENTS
        
        sp = 0.00005;    // variance for phase progression
        sv = 0.001;     // variance for phase speed progression
        ss = 0.00001;    // variance for scaling progression
        sr = 0.000000;  // variance for rotation progression
        
        coefficients.phaseVariance    = sp;
        coefficients.speedVariance    = sv;
        coefficients.scaleVariance    = ss;
        coefficients.rotationVariance = sr;
        
        
        // BUILDING OBJECT
        bubi = new GVF();
        bubi->setup(parameters, coefficients);
        
        
        restarted_l=1;
        restarted_d=1;
        
        refmap = new map<int, vector< pair<float,float> > >;
        
        state = STATE_CLEAR;
        
        lastreferencelearned = -1;
        currentToBeLearned = -1;
        value_mmax = -INFINITY;
        
        toBeTranslated = 1;
        offline_recognition = false;
		
    }
    
    ~gvf()
    {
        //post("destroying gvf object");
		if(bubi != NULL)
			delete bubi;
    }
    
	
	// methods:
	void bang(long inlet) {
		outlet_bang(m_outlets[0]);
	}
	void testfloat(long inlet, double v) {
        post(" float");
		outlet_float(m_outlets[0], v);
	}
	void testint(long inlet, long v) {
        post(" long");
		outlet_int(m_outlets[0], v);
	}
    
    
    
    
    ///////////////////////////////////////////////////////////
    //====================== LEARN
    ///////////////////////////////////////////////////////////
	void learn(long inlet, t_symbol * s, long ac, t_atom * av) {
        
        if(ac != 1)
        {
            post("wrong number of argument (must be 1)");
            return;
        }
        
        int refI = atom_getlong(&av[0]);
        
        refI = refI-1; // start at 1 in the patch but 0 in C++
        
        if(refI > lastreferencelearned+1)
        {
            post("you need to learn reference %d first",lastreferencelearned+1);
            return;
        }
        else{
            post("refI=%i, lastbla=%i", refI, lastreferencelearned);
            if(refI == lastreferencelearned+1)
            {
                lastreferencelearned++;
                //refmap[refI] = vector<pair<float, float> >();
                post("learning reference %d", refI+1);
                bubi->addTemplate();
            }
            else{
                //refmap[refI] = vector<pair<float, float> >();
                post("modifying reference %d", refI+1);
                bubi->clearTemplate(refI);
            }
        }
        
        state = STATE_LEARNING;
        
        restarted_l=1;
        
    }
    
    
    
    ///////////////////////////////////////////////////////////
    //====================== FOLLOW
    ///////////////////////////////////////////////////////////
    void follow(long inlet, t_symbol * s, long ac, t_atom * av) {
        
        if(lastreferencelearned >= 0)
        {
            post("I'm about to follow! restarting");

            if (state == STATE_LEARNING)
                bubi->learn();
            
            //bubi->spreadParticles(means,ranges);
            bubi->restart();
            
            restarted_l=1;
            restarted_d=1;
            //post("nb of gest after following %i", bubi->getNbOfGestures());
            
            state = STATE_FOLLOWING;
            
            
        }
        else
        {
            post("no reference has been learned");
            return;
        }
    }
    
    
    
    ///////////////////////////////////////////////////////////
    //====================== DATA
    ///////////////////////////////////////////////////////////
    void data(long inlet, t_symbol * s, long ac, t_atom * av) {
        /*
        if (ac!=inputDim){
            post("RESET w/ NEW DIM");
            if (bubi!=NULL)
                free(bubi);
            inputDim = ac;
            //bubi = new GVF(inputDim, ns, sigs, smoothingCoef, rt, 0.);
            //post("HERE");
            bubi = new GVF();
            parameters.inputDimensions=ac;
            bubi->setup(parameters, coefficients);
        }*/
        inputDim = ac;
        
        if(ac == 0)
        {
            post("invalid format, points have at least 1 coordinate");
            return;
        }
        
        // INCOMING DATA - CLEAR MODE
        // -----------------------------
        if(state == STATE_CLEAR)
        {
            post("I'm in a standby (must learn something beforehand)");
            return;
        }
        
        // INCOMING DATA - LEARNING MODE
        // -----------------------------
        else if(state == STATE_LEARNING)
        {
            
            vector<float> vect(inputDim);
            for (int k=0; k<ac; k++)
                vect[k] = atom_getfloat(av + k);
            
            // Fill template with the observation
            //            post("%i %f %f %f",bubi->getNumberOfTemplates(),vect[0],vect[1],vect[2]);
            bubi->fillTemplate(bubi->getNumberOfTemplates()-1,vect);
            
        }
        
        // INCOMING DATA - FOLLOWING MODE
        // ------------------------------
        else if(state == STATE_FOLLOWING)
        {
                  //  post("dimension %i", inputDim);
            
            vector<float> vect(inputDim);
            for (int k=0; k<ac; k++)
                vect[k] = atom_getfloat(av + k);
            
            // Perform the inference with the current observation
           
            bubi->infer(vect);
           
           
            
            // OUTPUT ESTIMATION OF VARIATIONS
            
            int nboftemplates = bubi->getNumberOfTemplates();
            vector<vector<float> > statu = bubi->getEstimatedStatus(); //getGestureProbabilities();
            t_atom *outAtoms = new t_atom[nboftemplates];
            
            // => PHASE
            for(int j = 0; j < nboftemplates; j++)
                atom_setfloat(&outAtoms[j],statu[j][0]);
            outlet_anything(m_outlets[0], gensym("phase"), nboftemplates, outAtoms);
            delete[] outAtoms;
            
            // => SPEED
            for(int j = 0; j < nboftemplates; j++)
                atom_setfloat(&outAtoms[j],statu[j][1]);
            outlet_anything(m_outlets[0], gensym("speed"), nboftemplates, outAtoms);
            delete[] outAtoms;
            
            
            
            int scaleCoefficients = bubi->getScaleInitialSpreading().size();
            int numberRotationAngles = bubi->getRotationInitialSpreading().size();
            
            // => SCALE
            outAtoms = new t_atom[nboftemplates * scaleCoefficients];
            for(int j = 0; j < nboftemplates; j++)
                for(int k = 0; k < scaleCoefficients; k++)
                    atom_setfloat(&outAtoms[j * scaleCoefficients + k],statu[j][2+k]);
            
            outlet_anything(m_outlets[0], gensym("scale"), nboftemplates * scaleCoefficients, outAtoms);
            delete[] outAtoms;
            
            
            // => ROTATION
            outAtoms = new t_atom[nboftemplates * numberRotationAngles];
            for(int j = 0; j < nboftemplates; j++)
                for(int k = 0; k <numberRotationAngles; k++)
                    atom_setfloat(&outAtoms[j * numberRotationAngles + k],statu[j][2+scaleCoefficients+k]);
            outlet_anything(m_outlets[0], gensym("angle"), nboftemplates * numberRotationAngles, outAtoms);
            delete[] outAtoms;
            
            
            
            
            
            // OUTPUT GESTURE PROBABILITIES (RECOGNITION)
            
            vector<float> gprob = bubi->getGestureProbabilities();
            
            outAtoms = new t_atom[nboftemplates];
            for(int j = 0; j < nboftemplates; j++)
                atom_setfloat(&outAtoms[j],gprob[j]);
            outlet_anything(m_outlets[1], gensym("weights"), nboftemplates, outAtoms);
            delete[] outAtoms;
            
            
             std::vector<float> aw = bubi->getAbsoluteProbabilities();
             outAtoms = new t_atom[aw.size()];
             for(int j = 0; j < aw.size(); j++)
                 atom_setfloat(&outAtoms[j],aw[j]);
             outlet_anything(m_outlets[2], gensym("absweights"), aw.size(), outAtoms);
             delete[] outAtoms;
             
             
             
             
            
            /*
             std::vector<float>* offs = bubi->getOffsets();
             outAtoms = new t_atom[(*offs).size()];
             for(int j = 0; j < (*offs).size(); j++)
             atom_setfloat(&outAtoms[j],(*offs)[j]);
             outlet_anything(m_outlets[2], gensym("offset"), (*offs).size(), outAtoms);
             delete[] outAtoms;
             */
             
            
        }
    }
    
    
    
    
    ///////////////////////////////////////////////////////////
    //====================== SAVE_VOCABULARY
    ///////////////////////////////////////////////////////////
    void save_vocabulary(long inlet, t_symbol * s, long ac, t_atom * av){
        char* mpath = atom_string(av);
        //        string filename = "/Users/caramiaux/gotest";
        int i=0;
        while ( *(mpath+i)!='/' )
            i++;
        mpath = mpath+i;
        string filename(mpath);

        bubi->saveTemplates(filename);
        
    }
    
    
    
    ///////////////////////////////////////////////////////////
    //====================== LOAD_VOCABULARY
    ///////////////////////////////////////////////////////////
    void load_vocabulary(long inlet, t_symbol * s, long ac, t_atom * av){
        char* mpath = atom_string(av);
        //        string filename = "/Users/caramiaux/gotest.txt";
        int i=0;
        while ( *(mpath+i)!='/' )
            i++;
        mpath = mpath+i;
        string filename(mpath);
        bubi->loadTemplates(filename);
        lastreferencelearned=bubi->getNumberOfTemplates()-1;
        
        t_atom* outAtoms = new t_atom[1];
        atom_setlong(&outAtoms[0],bubi->getNumberOfTemplates());
        outlet_anything(m_outlets[2], gensym("vocabulary_size"), 1, outAtoms);
        delete[] outAtoms;
    }
    
    
    
    ///////////////////////////////////////////////////////////
    //====================== CLEAR
    ///////////////////////////////////////////////////////////
    void clear(long inlet, t_symbol * s, long ac, t_atom * av) {
        lastreferencelearned = -1;
        
        bubi->clear();
        
        restarted_l=1;
        restarted_d=1;
        
        value_mmax = -INFINITY;
        
        state = STATE_CLEAR;
    }
    
    
    
    ///////////////////////////////////////////////////////////
    //====================== PRINTME
    ///////////////////////////////////////////////////////////
    void printme(long inlet, t_symbol * s, long ac, t_atom * av) {
        post("********** parameters **********");
        post("[Number of particles] %d ", bubi->getNumberOfParticles());
        post("[Resampling Threshold] %d ", bubi->getResamplingThreshold());
        post("[Tolerance] %.2f ", bubi->getParameters().tolerance);
        post("[Feature variances] %.6f %.6f %.6f %.6f: ", bubi->getVarianceCoefficents().phaseVariance,bubi->getVarianceCoefficents().speedVariance,bubi->getVarianceCoefficents().scaleVariance,bubi->getVarianceCoefficents().rotationVariance);
        
        //vector<float> means = bubi->getInitia
        post("[Spreading] phase: %.3f speed: %.3f num-scale-coef: %i num-angles: %i",
             bubi->getPhaseInitialSpreading(),
             bubi->getSpeedInitialSpreading(),
             bubi->getScaleInitialSpreading().size(),
             bubi->getRotationInitialSpreading().size());
        //post("%i %i",means.size(),ranges.size());
        //post("Means: %.3f %.3f %.3f %.3f", means[0], means[1], means[2], means[3]);
        //post("Ranges: %.3f %.3f %.3f %.3f", ranges[0], ranges[1], ranges[2], ranges[3]);
        post("********** vocabulary **********");
        post("Number of templates: %d", bubi->getNumberOfTemplates());
        for(int i = 0; i < bubi->getNumberOfTemplates(); i++)
        {
            post("reference: %d [length: %d]", i+1, bubi->getLengthOfTemplateByIndex(i));
            vector<vector<float> > tplt = bubi->getTemplateByIndex(i);
            for(int j = 0; j < tplt.size(); j++)
            {
                if (inputDim==2)
                    post("%02.4f  %02.4f", tplt[j][0], tplt[j][1]);
                if (inputDim==3)
                    post("%02.4f  %02.4f  %02.4f", tplt[j][0], tplt[j][1], tplt[j][2]);
            }
        }
        
    }
    
    
    
    ///////////////////////////////////////////////////////////
    //====================== RESTART
    ///////////////////////////////////////////////////////////
    void restart(long inlet, t_symbol * s, long ac, t_atom * av) {
        restarted_l=1;
        if(state == STATE_FOLLOWING)
        {
            
            //bubi->spreadParticles(means,ranges);
            //post("commenting restart");
            bubi->restart();
            
            restarted_l=1;
            restarted_d=1;
        }
    }
    
    
    
    
    ///////////////////////////////////////////////////////////
    //====================== tolerance
    ///////////////////////////////////////////////////////////
    void tolerance(long inlet, t_symbol * s, long ac, t_atom * av) {
        float stdnew = atom_getfloat(&av[0]);
        if (stdnew == 0.0)
            stdnew = 0.1;
        bubi->setTolerance(stdnew);
    }
    
    ///////////////////////////////////////////////////////////
    //====================== resampling_threshold
    ///////////////////////////////////////////////////////////
    void resampling_threshold(long inlet, t_symbol * s, long ac, t_atom * av) {
        int rtnew = atom_getlong(&av[0]);
        int cNS = bubi->getNumberOfParticles();
        if (rtnew >= cNS)
            rtnew = floor(cNS/2);
        bubi->setResamplingThreshold(rtnew);
    }
    
    ///////////////////////////////////////////////////////////
    //====================== spreading_means
    ///////////////////////////////////////////////////////////
    void spreading_means(long inlet, t_symbol * s, long ac, t_atom * av) {
        //        means = Eigen::VectorXf(pdim);
        for (int k=0;k<pdim;k++)
            means[k] = atom_getfloat(&av[k]); // atom_getfloat(&av[1]), atom_getfloat(&av[2]), atom_getfloat(&av[3]);
    }
    
    ///////////////////////////////////////////////////////////
    //====================== spreading_ranges
    ///////////////////////////////////////////////////////////
    void spreading_ranges(long inlet, t_symbol * s, long ac, t_atom * av) {
        //        means = Eigen::VectorXf(pdim);
        for (int k=0;k<pdim;k++)
            ranges[k] = atom_getfloat(&av[k]);
        //ranges = Eigen::VectorXf(pdim);
        //ranges << atom_getfloat(&av[0]), atom_getfloat(&av[1]), atom_getfloat(&av[2]), atom_getfloat(&av[3]);
    }
    
    ///////////////////////////////////////////////////////////
    //====================== adaptation_speed
    ///////////////////////////////////////////////////////////
    void adaptation_speed(long inlet, t_symbol * s, long ac, t_atom * av) {
        /*vector<float> as;
         as.push_back(atom_getfloat(&av[0]));
         as.push_back(atom_getfloat(&av[1]));
         as.push_back(atom_getfloat(&av[2]));
         as.push_back(atom_getfloat(&av[3]));*/
        
        coefficients.phaseVariance=atom_getfloat(&av[0]);
        coefficients.speedVariance=atom_getfloat(&av[1]);
        coefficients.scaleVariance=atom_getfloat(&av[2]);
        coefficients.rotationVariance=atom_getfloat(&av[3]);
        
        bubi->setVarianceCoefficents(coefficients);
    }
    
    /*    void probThresh(long inlet, t_symbol * s, long ac, t_atom * av) {
     bubi->probThresh = atom_getlong(&av[0])*bubi->getNbOfParticles();
     }
     
     void probThreshMin(long inlet, t_symbol * s, long ac, t_atom * av) {
     bubi->probThreshMin = atom_getlong(&av[0])*bubi->getNbOfParticles();
     }*/
    
    ///////////////////////////////////////////////////////////
    //====================== translate
    ///////////////////////////////////////////////////////////
    void translate(long inlet, t_symbol * s, long ac, t_atom * av){
        toBeTranslated = atom_getlong(&av[0]);
        if (toBeTranslated==1) parameters.translate=true;
        else parameters.translate=false;
        bubi->setParameters(parameters);
    }
    
};


//THIS IS FOR Max6.1

//C74_EXPORT extern "C" int main(void) {
//	// create a class with the given name:
//	gvf::makeMaxClass("gvf");
//	REGISTER_METHOD(gvf, bang);
//	REGISTER_METHOD_FLOAT(gvf, testfloat);
//	REGISTER_METHOD_LONG(gvf, testint);
//	REGISTER_METHOD_GIMME(gvf, test);
//}


extern "C" int main(void) {
    // create a class with the given name:
    gvf::makeMaxClass("gvf");
    REGISTER_METHOD(gvf, bang);
    REGISTER_METHOD_FLOAT(gvf, testfloat);
    REGISTER_METHOD_LONG(gvf,  testint);
    REGISTER_METHOD_GIMME(gvf, learn);
    REGISTER_METHOD_GIMME(gvf, follow);
    REGISTER_METHOD_GIMME(gvf, clear);
    REGISTER_METHOD_GIMME(gvf, data);
    REGISTER_METHOD_GIMME(gvf, printme);
    REGISTER_METHOD_GIMME(gvf, restart);
    REGISTER_METHOD_GIMME(gvf, tolerance);
    REGISTER_METHOD_GIMME(gvf, resampling_threshold);
    REGISTER_METHOD_GIMME(gvf, spreading_means);
    REGISTER_METHOD_GIMME(gvf, spreading_ranges);
    REGISTER_METHOD_GIMME(gvf, adaptation_speed);
    REGISTER_METHOD_GIMME(gvf, save_vocabulary);
    REGISTER_METHOD_GIMME(gvf, load_vocabulary);
    REGISTER_METHOD_GIMME(gvf, translate);
}

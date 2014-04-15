
///////////////////////////////////////////////////////////////////////
//
//  GVF - Gesture Variation Follower PureData Object
//
//
//  Copyright (C) 2013 Baptiste Caramiaux, Goldsmiths College, University of London
//
//  The GVF library is under the GNU Lesser General Public License (LGPL v3)
//  version: 04-2014
//
///////////////////////////////////////////////////////////////////////


//#include <boost/random.hpp>
#include <vector>
#include "m_pd.h"
#include "GVF.h"
#include "globalutilities.h"

#include <iostream>
#include <fstream>
#include <string>

static t_class *gvf_class;


typedef struct _gvf {
    t_object  x_obj;
    GVF *bubi;
	t_int state;
	t_int lastreferencelearned;
    t_int currentToBeLearned;
    std::map<int,std::vector<std::pair<float,float> > > *refmap;
	t_int ns, rt;
	t_float sp, sv, sr, ss, smoothingCoef; // pos,vel,rot,scal,observation
	t_int pdim;

    int translate;
    int inputDim;
    // outlets
    t_outlet *Position,*Vitesse,*Scaling,*Rotation,*Recognition,*Likelihoods,*Number_templates;

    GVF::GVFVarianceCoefficents coefficients;
    GVF::GVFParameters parameters;

} t_gvf;

static void gvf_learn      (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_follow     (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_clear      (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_data       (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_printme    (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_restart    (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_std        (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_rt         (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_means      (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_ranges     (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);
static void gvf_adaptspeed (t_gvf *x,const t_symbol *sss,int argc, t_atom *argv);

static void gvf_auto(t_gvf *x);

t_int restarted_l;
t_int restarted_d;
std::pair<t_float,t_float> xy0_l;
std::pair<t_float,t_float> xy0_d;
std::vector<t_float> vect_0_l;
std::vector<t_float> vect_0_d;
enum {STATE_CLEAR, STATE_LEARNING, STATE_FOLLOWING};


static void *gvf_new(t_symbol *s, int argc, t_atom *argv)
{
    post("\nGVF - realtime adaptive gesture recognition (version: 15-04-2014)");
    post("(c) Goldsmiths, University of London and Ircam - Centre Pompidou");
    
    t_gvf *x = (t_gvf *)pd_new(gvf_class);
    
    // PARAMETERS

    x->inputDim = 3;   // input dimensions (default: 2-d gestures)
    x->ns = 2000;      // number of particles
    x->rt = 500;       // resampling threshold
    x->smoothingCoef = 0.5; // smoothing coef (or tolerance)
        
    x->parameters.inputDimensions = x->inputDim;
    x->parameters.numberParticles = x->ns;
    x->parameters.tolerance = x->smoothingCoef;
    x->parameters.resamplingThreshold = x->rt;
    x->parameters.distribution = 0.0f;
    x->parameters.translate = true;
    x->parameters.allowSegmentation = false;

    // COEFFICIENTS
    
    x->sp = 0.00005;
    x->sv = 0.001;
    x->ss = 0.0001;
    x->sr = 0.0000001;
    
    x->coefficients.phaseVariance    = x->sp;
    x->coefficients.speedVariance    = x->sv;
    x->coefficients.scaleVariance    = x->ss;
    x->coefficients.rotationVariance = x->sr;
    
    // BUILDING OBJECT
    x->bubi = new GVF();
    x->bubi->setup(x->parameters, x->coefficients);

    
    restarted_l=1;
    restarted_d=1;
    
    x->state = STATE_CLEAR;
    x->lastreferencelearned = -1;
    x->currentToBeLearned = -1;
   
    if (argc > 0)
        x->ns = getint(argv);
    if (argc > 1)
        x->rt = getint(argv + 1);
    
    x->Position     = outlet_new(&x->x_obj, &s_list);
    x->Vitesse      = outlet_new(&x->x_obj, &s_list);
    x->Rotation     = outlet_new(&x->x_obj, &s_list);
    x->Scaling      = outlet_new(&x->x_obj, &s_list);
    x->Recognition  = outlet_new(&x->x_obj, &s_list);
    x->Likelihoods  = outlet_new(&x->x_obj, &s_list);
    x->Number_templates= outlet_new(&x->x_obj, &s_list);

    return (void *)x;
}

static void gvf_destructor(t_gvf *x)
{
    //post("gvf destructor...");
    if(x->bubi != NULL)
        delete x->bubi;
    if(x->refmap != NULL)
        delete x->refmap;
    //post("destructor complete");
}




///////////////////////////////////////////////////////////
//====================== LEARN
///////////////////////////////////////////////////////////
static void gvf_learn(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    if(argc != 1)
    {
        post("need another argument");
        return;
    }
    int refI = getint(argv);
    refI=refI-1; // starts at 1 in the patch but 0 in C++
    x->currentToBeLearned=refI;
    if(refI > x->lastreferencelearned+1)
    {
        post("you need to learn reference %d first",x->lastreferencelearned+1);
        return;
    }
    else{
        if(refI == x->lastreferencelearned+1)
            {
                x->lastreferencelearned++;
                //refmap[refI] = vector<pair<float, float> >();
                post("learning reference %d", refI+1);
                x->bubi->addTemplate();
            }
            else{
                //refmap[refI] = vector<pair<float, float> >();
                post("modifying reference %d", refI+1);
                x->bubi->clearTemplate(refI);
            }
        }
    //x->lastreferencelearned++;
    //x->refmap[refI];// = std::vector<std::pair<float, float> >();
    //(*x->refmap)[refI] = std::vector<std::pair<float, float> >();
    //post("learning reference %d", refI+1);
    //x->bubi->addTemplate();
    
    x->state = STATE_LEARNING;
    
    restarted_l=1;
    
    
    // output number of templates
    t_atom *outAtoms = new t_atom[1];
    SETFLOAT(&outAtoms[0],x->lastreferencelearned+1);
    outlet_list(x->Number_templates, &s_list, 1, outAtoms);
    delete[] outAtoms;
}


///////////////////////////////////////////////////////////
//====================== FOLLOW
///////////////////////////////////////////////////////////
static void gvf_follow(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    if(x->lastreferencelearned >= 0)
    {
        
        if (x->state == STATE_LEARNING)
            x->bubi->learn();

        //x->bubi->spreadParticles(x->mpvrs,x->rpvrs);
        x->bubi->restart();
            
        restarted_l=1;
        restarted_d=1;

        x->state = STATE_FOLLOWING;
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
static void gvf_data(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    
    x->inputDim = argc;

    if(x->state == STATE_CLEAR)
    {
        post("what am I supposed to do ... I'm in standby!");
        return;
    }
    if(argc == 0)
    {
        post("invalid format, points have at least 1 coordinate");
        return;
    }
    if(x->state == STATE_LEARNING)
    {
        std::vector<float> vect(argc);
         /*       
	if (x->translate){
            if (restarted_l==1)
            {
                // store the incoming list as a vector of float
                for (int k=0; k<argc; k++){
                    //vect[k] = GetFloat(argv[k]);
                    //may not be correct..
                    vect[k] = getfloat(argv + k);
                }
                // keep track of the first point
                vect_0_l = vect;
                restarted_l=0;
            }
            for (int k=0; k<argc; k++){
                // repeat of above, could remove
                vect[k] = getfloat(argv + k);
                vect[k] = vect[k]-vect_0_l[k];
            }
        }
        else {
            for (int k=0; k<argc; k++)
                vect[k] = getfloat(argv + k);
        }*/

        for (int k=0; k<argc; k++)
                vect[k] = getfloat(argv + k);
        
        //				pair<float,float> xy;
        //				xy.first  = x -xy0_l.first;
        //				xy.second = y -xy0_l.second;
        
        // Fill template
        //x->bubi->fillTemplate(x->lastreferencelearned,vect);
        x->bubi->fillTemplate(x->currentToBeLearned,vect);
        
    }
    else if(x->state == STATE_FOLLOWING)
    {
        std::vector<float> vect(argc);
/*
        if (x->translate){
            if (restarted_d==1)
            {
                // store the incoming list as a vector of float
                for (int k=0; k<argc; k++){
                    vect[k] = getfloat(argv + k);
                }
                // keep track of the first point
                vect_0_d = vect;
                restarted_d=0;
            }
            for (int k=0; k<argc; k++){
                //?
                vect[k] = vect[k] = getfloat(argv + k);
                vect[k] = vect[k] - vect_0_d[k];
            }
        }
        else{
            for (int k=0; k<argc; k++)
                vect[k] = vect[k] = getfloat(argv + k);
        }*/
        
        for (int k=0; k<argc; k++)
                vect[k] = vect[k] = getfloat(argv + k);
        
        //post("%f %f",xy(0,0),xy(0,1));
        // ------- Fill template
        x->bubi->infer(vect);


        // OUTPUTS
        int nboftemplates = x->bubi->getNumberOfTemplates();
        vector<vector<float> > statu = x->bubi->getEstimatedStatus(); //getGestureProbabilities();
        t_atom *outAtoms = new t_atom[nboftemplates];
            
        
        // --- Phase
        for(int j = 0; j < nboftemplates; j++)
            SETFLOAT(&outAtoms[j],statu[j][0]);
        outlet_list(x->Position, &s_list, nboftemplates, outAtoms);
        delete[] outAtoms;
        
        // --- Speed
        for(int j = 0; j < nboftemplates; j++)
            SETFLOAT(&outAtoms[j],statu[j][1]);
        outlet_list(x->Vitesse, &s_list, nboftemplates, outAtoms);
        delete[] outAtoms;
        
        
        int scaleCoefficients = x->bubi->getScaleInitialSpreading().size();
        int numberRotationAngles = x->bubi->getRotationInitialSpreading().size();

        // --- Scale
        outAtoms = new t_atom[nboftemplates * scaleCoefficients];
        for(int j = 0; j < nboftemplates; j++)
            for(int k = 0; k < scaleCoefficients; k++)
                    SETFLOAT(&outAtoms[j * scaleCoefficients + k],statu[j][2+k]);
        outlet_list(x->Rotation, &s_list, nboftemplates, outAtoms);
        delete[] outAtoms;
        
        // --- Orientation
        outAtoms = new t_atom[nboftemplates * numberRotationAngles];
        for(int j = 0; j < nboftemplates; j++)
            for(int k = 0; k <numberRotationAngles; k++)
                SETFLOAT(&outAtoms[j * numberRotationAngles + k],statu[j][2+scaleCoefficients+k]);
        outlet_list(x->Scaling, &s_list, nboftemplates, outAtoms);
        delete[] outAtoms;
        
        

        // -- Probabilities

        vector<float> gprob = x->bubi->getGestureProbabilities();

        outAtoms = new t_atom[nboftemplates];
        for(int j = 0; j < nboftemplates; j++)
            SETFLOAT(&outAtoms[j],gprob[j]);
        outlet_list(x->Likelihoods, &s_list, nboftemplates, outAtoms);
        delete[] outAtoms;

/*
        float probmaxsofar=-1;
        int probmaxsofarindex=-1;
        for (int k=0; k<gprob.size(); k++){
            if (gprob(k)>probmaxsofar){
                probmaxsofar=gprob(k);
                probmaxsofarindex=k+1;
            }
        }
        
        outAtoms = new t_atom[1];
        SETFLOAT(&outAtoms[0],probmaxsofarindex);
        outlet_list(x->Recognition, &s_list, probmaxsofarindex, outAtoms);
        delete[] outAtoms;
        
        
        outAtoms = new t_atom[gprob.size()];
        for(int j = 0; j < gprob.size(); j++)
            SETFLOAT(&outAtoms[j],gprob(j,0));
        outlet_list(x->Likelihoods, &s_list, gprob.size(), outAtoms);
        delete[] outAtoms;*/
    
        
        
        
    }
}


///////////////////////////////////////////////////////////
//====================== SAVE_VOCABULARY
///////////////////////////////////////////////////////////
static void gvf_save_vocabulary(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    unsigned int bufsize = 200;
    char * mpath;
    mpath = (char*)malloc(bufsize*sizeof(bufsize));
    atom_string(argv, mpath, bufsize);
    int i=0;
    while ( *(mpath+i)!='/' )
        i++;
    mpath = mpath+i;
    std::string filename(mpath);
    x->bubi->saveTemplates(filename);
}



///////////////////////////////////////////////////////////
//====================== LOAD_VOCABULARY
///////////////////////////////////////////////////////////
void gvf_load_vocabulary(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    unsigned int bufsize = 200;
    char * mpath;
    mpath = (char*)malloc(bufsize*sizeof(bufsize));
    atom_string(argv, mpath, bufsize);
    int i=0;
    while ( *(mpath+i)!='/' )
        i++;
    mpath = mpath+i;
    std::string filename(mpath);
    //std::string filename = "/Users/caramiaux/gotest.txt";
    x->bubi->loadTemplates(filename);
    x->lastreferencelearned=x->bubi->getNumberOfTemplates()-1;
/*
    t_atom* outAtoms = new t_atom[1];
    SETFLOAT(&outAtoms[0], x->bubi->getNbOfTemplates());
    outlet_list(x->Likelihoods, &s_list, 1, outAtoms);
    delete[] outAtoms;
 */
    
}


///////////////////////////////////////////////////////////
//====================== CLEAR
///////////////////////////////////////////////////////////
static void gvf_clear(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    x->lastreferencelearned = -1;
    x->bubi->clear();
    restarted_l=1;
    restarted_d=1;
    x->state = STATE_CLEAR;
}


///////////////////////////////////////////////////////////
//====================== PRINTME
///////////////////////////////////////////////////////////
static void gvf_printme(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    post("********** parameters **********");
    post("[Number of particles] %d ", x->bubi->getNumberOfParticles());
    post("[Resampling Threshold] %d ", x->bubi->getResamplingThreshold());
        post("[Tolerance] %.2f ", x->bubi->getParameters().tolerance);
        post("[Feature variances] %.6f %.6f %.6f %.6f: ", 
            x->bubi->getVarianceCoefficents().phaseVariance,
            x->bubi->getVarianceCoefficents().speedVariance,
            x->bubi->getVarianceCoefficents().scaleVariance,
            x->bubi->getVarianceCoefficents().rotationVariance);
        
        //vector<float> means = bubi->getInitia
        post("[Spreading] phase: %.3f speed: %.3f num-scale-coef: %i num-angles: %i",
             x->bubi->getPhaseInitialSpreading(),
             x->bubi->getSpeedInitialSpreading(),
             x->bubi->getScaleInitialSpreading().size(),
             x->bubi->getRotationInitialSpreading().size());
        
        post("********** vocabulary **********");
        post("Number of templates: %d", x->bubi->getNumberOfTemplates());
        for(int i = 0; i < x->bubi->getNumberOfTemplates(); i++)
        {
            post("reference: %d [length: %d]", i+1, x->bubi->getLengthOfTemplateByIndex(i));
            vector<vector<float> > tplt = x->bubi->getTemplateByIndex(i);
            for(int j = 0; j < tplt.size(); j++)
            {
                if (x->inputDim==2)
                    post("%02.4f  %02.4f", tplt[j][0], tplt[j][1]);
                if (x->inputDim==3)
                    post("%02.4f  %02.4f  %02.4f", tplt[j][0], tplt[j][1], tplt[j][2]);
            }
        }
        
}


///////////////////////////////////////////////////////////
//====================== RESTART
///////////////////////////////////////////////////////////
static void gvf_restart(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    restarted_l=1;
    if(x->state == STATE_FOLLOWING)
    {
        //x->bubi->spreadParticles(x->mpvrs,x->rpvrs);
        x->bubi->restart();
        restarted_l=1;
        restarted_d=1;
    }

}


///////////////////////////////////////////////////////////
//====================== tolerance
///////////////////////////////////////////////////////////
static void gvf_tolerance(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    float stdnew = getfloat(argv);
    if (stdnew == 0.0)
        stdnew = 0.1;
    //x->bubi->setIcovSingleValue(1/(stdnew*stdnew));
    x->bubi->setTolerance(stdnew);
}


///////////////////////////////////////////////////////////
//====================== resampling_threshold
///////////////////////////////////////////////////////////
static void gvf_resampling_threshold(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    int rtnew = getint(argv);
    int cNS = x->bubi->getNumberOfParticles();
    if (rtnew >= cNS)
        rtnew = floor(cNS/2);
    x->bubi->setResamplingThreshold(rtnew);
}


///////////////////////////////////////////////////////////
//====================== spreading_means
///////////////////////////////////////////////////////////
static void gvf_spreading_means(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    post("TO-BE-TESTED");
    //x->mpvrs = Eigen::VectorXf(x->pdim);
    //x->mpvrs << getfloat(argv), getfloat(argv + 1), getfloat(argv + 2), getfloat(argv + 3);
}


///////////////////////////////////////////////////////////
//====================== spreading_ranges
///////////////////////////////////////////////////////////
static void gvf_spreading_ranges(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    post("TO-BE-TESTED");
    //x->rpvrs = Eigen::VectorXf(x->pdim);
    //x->rpvrs << getfloat(argv), getfloat(argv + 1), getfloat(argv + 2), getfloat(argv + 3);
}


///////////////////////////////////////////////////////////
//====================== adaptation_speed
///////////////////////////////////////////////////////////
static void gvf_adaptation_speed(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    /*
    std::vector<float> as;
    as.push_back(getfloat(argv));
    as.push_back(getfloat(argv + 1));
    as.push_back(getfloat(argv + 2));
    as.push_back(getfloat(argv + 3));
    x->bubi->setAdaptSpeed(as);
    */

    x->coefficients.phaseVariance=getfloat(&argv[0]);
    x->coefficients.speedVariance=getfloat(&argv[1]);
    x->coefficients.scaleVariance=getfloat(&argv[2]);
    x->coefficients.rotationVariance=getfloat(&argv[3]);
        
    x->bubi->setVarianceCoefficents(x->coefficients);
}

///////////////////////////////////////////////////////////
//====================== translate
///////////////////////////////////////////////////////////
static void gvf_translate(t_gvf *x,const t_symbol *sss,int argc, t_atom *argv)
{
    int toBeTranslated = getint(&argv[0]);
    if (toBeTranslated==1) x->parameters.translate=true;
    else x->parameters.translate=false;
    x->bubi->setParameters(x->parameters);
    //post("transalte %i", getint(argv));
    //x->translate = getint(argv);
}

extern "C"
{
    void gvf_setup(void) {
        gvf_class = class_new( gensym("gvf"),(t_newmethod)gvf_new,(t_method)gvf_destructor,
                                sizeof(t_gvf),CLASS_DEFAULT,A_GIMME,0);
        
        class_addmethod(gvf_class,(t_method)gvf_learn,gensym("learn"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_follow,gensym("follow"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_clear,gensym("clear"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_data,gensym("data"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_printme,gensym("printme"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_restart,gensym("restart"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_tolerance,gensym("tolerance"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_resampling_threshold,gensym("resampling_threshold"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_spreading_means,gensym("spreading_means"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_spreading_ranges,gensym("spreading_ranges"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_adaptation_speed,gensym("adaptation_speed"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_save_vocabulary,gensym("save_vocabulary"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_load_vocabulary,gensym("load_vocabulary"),A_GIMME,0);
        class_addmethod(gvf_class,(t_method)gvf_translate,gensym("translate"),A_GIMME,0);
    }
}

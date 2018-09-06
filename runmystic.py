import time
import numpy as np
import h5py
import os
from joblib import Parallel, delayed
num_cores=5*2
fullRun=True
model='m05'
if fullRun:
    mu0=np.arange(0.05,1.05,0.05)
    mu=np.arange(0.05,1.05,0.05)
    az=np.arange(0.0,189.0,9.0)
    wav=['466','554','412','442','487','530','547']
    aotx=['01','02','03','04','05','06','07','08','09','10','11','12','13']
else:
    mu0=np.array([0.1,0.3,0.5,0.7,0.9,1.0])
    mu=np.array([0.05,0.1,0.3,0.5,0.7,0.9,1.0])
    az=np.array([0.0,45.0,90.0,135.0,180.0])
    wav=['466','412','547'] 
    aotx=['01','02','03','04','05','13']
wav_total=['466','554','412','442','487','530','547'] # sequence is IMPORTANT when LUTs are combined
mysticExtensions=['.rad.spc','.rad.std.spc','.rad','2.rad','.rad.std','2.rad.std','.flx','2.flx','.flx.spc','.flx.std','2.flx.std','.flx.std.spc']
sz=np.arccos(mu0)*180.0/np.pi
nwav=len(wav)
nwav_total=len(wav_total)
nsz=len(sz)
nmu=len(mu)
naz=len(az)
naot=len(aotx)
fname_mystic_const='mystic_const.inp'
def runMysticByAngles(isz,iaz,imu):
    vLut_rad=np.zeros((naot,nwav))
    vLut_std=np.zeros((naot,nwav))
    fname_mystic_run='../examples/inp/isz'+str(isz)+'iaz'+str(iaz)+'imu'+str(imu)+'.inp'
    faotssa_txt=model+'_wav_aot_ssa.txt'
    fdat=np.loadtxt('../examples/models/'+model+'/'+faotssa_txt)
    aot_ssa=fdat[:,1:3]
    fin_const=open(fname_mystic_const).read()
    for iaot in range(naot):
        for iwav in range(nwav):
            basename='../examples/out/isz'+str(isz)+'iaz'+str(iaz)+'imu'+str(imu)
            wav_nm=float(wav[iwav])
            fname_gas=wav[iwav]+'.gas'
            fname_ray=wav[iwav]+'.ray'
            fname_pm=model+'_iaot'+aotx[iaot]+'_'+wav[iwav]+'.cdf'
            fpath_pm=model+'/'+fname_pm
            ix_aot_ssa=(int(aotx[iaot])-1)*nwav_total+wav_total.index(wav[iwav])
            aot=aot_ssa[ix_aot_ssa,0]
            ssa=aot_ssa[ix_aot_ssa,1]
            fin_run=open(fname_mystic_run,'w')
            fin_run.write(fin_const)
            fin_run.write('\n')
            fin_run.write('wavelength %s\n'%wav_nm)
            fin_run.write('wc_modify tau set %s\n'%aot)
            fin_run.write('wc_modify ssa set %s\n'%ssa)
            fin_run.write('wc_properties ../examples/models/%s interpolate\n'%fpath_pm)
            fin_run.write('mol_tau_file sca ../examples/raygas/%s\n'%fname_ray)
            fin_run.write('mol_tau_file abs ../examples/raygas/%s\n'%fname_gas)
            fin_run.write('sza %s\n'%sz[isz])
            fin_run.write('phi %s\n'%az[iaz])
            fin_run.write('umu %s\n'%mu[imu])
            fin_run.write('mc_basename %s\n'%basename)
            fin_run.write('quiet\n')
            fin_run.close()
	    #os.system('/home/skorkin/libradtran202gcc55/bin/uvspec < '+fname_mystic_run)
            fin,fout=os.popen2('/home/skorkin/libradtran202gcc55/bin/uvspec < '+fname_mystic_run)
            os.wait()
            vLut_rad[iaot,iwav]=np.pi*np.loadtxt(basename+'.rad.spc')[0, 4]
            vLut_std[iaot,iwav]=np.pi*np.loadtxt(basename+'.rad.std.spc')[0, 4]
    f=h5py.File(basename+'.h5','w')
    f['vLUT_rad']=vLut_rad.copy()
    f['vLUT_std']=vLut_std.copy()
    f.close()
    os.remove(fname_mystic_run)
    for ext in mysticExtensions:
        try:
            os.remove(basename+ext)
        except:
            pass
def combineMysticLUTbyAngles():
    vLut_rad=np.zeros((nsz,naz,nmu,naot,nwav))
    vLut_std=np.zeros((nsz,naz,nmu,naot,nwav))
    for isz in range(nsz):
        for iaz in range(naz):
            for imu in range(nmu):
                fPathName='../examples/out/isz'+str(isz)+'iaz'+str(iaz)+'imu'+str(imu)+'.h5'
                f=h5py.File(fPathName,'r')
                vLut_rad[isz,iaz,imu]=f['vLUT_rad'][:]
                vLut_std[isz,iaz,imu]=f['vLUT_std'][:]
                f.close()
                #os.remove(fPathName)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< H5_FILE_REMOVE
    out_rad=np.zeros((naot*nwav*nsz*naz+1,nmu+2))
    out_rad[0,0]=-999.0
    out_rad[0,1]=-999.0
    out_rad[0,2:nmu+3]=mu
    out_std=np.zeros((naot*nwav*nsz*naz+1,nmu+2))
    out_std[0,0]=-999.0
    out_std[0,1]=-999.0
    out_std[0,2:nmu+3]=mu
    ix=0
    for iaot in range(naot):
        for iwav in range(nwav):
            for isz in range(nsz):
                for iaz in range(naz):
                    ix+=1
                    out_rad[ix,0]=mu0[isz]
                    out_rad[ix,1]=az[iaz]
                    out_rad[ix,2:nmu+3]=vLut_rad[isz,iaz,0:nmu,iaot,iwav]
                    out_std[ix,0]=mu0[isz]
                    out_std[ix,1]=az[iaz]
                    out_std[ix,2:nmu+3]=vLut_std[isz,iaz,0:nmu,iaot,iwav]
    np.savetxt(model+'rad.out',out_rad,fmt=['%10.5f','%10.2f']+['%10.5f']*nmu)
    np.savetxt(model+'std.out',out_std,fmt=['%10.5f','%10.2f']+['%10.5f']*nmu)
if __name__=='__main__':
    t1_sec=time.time()
    Parallel(n_jobs = num_cores)(delayed(runMysticByAngles)(isz,iaz,imu) \
                 for isz in range(nsz) for iaz in range(naz) for imu in range(nmu))
    combineMysticLUTbyAngles()
    t2_sec=time.time()
    print 'total time, s.:',t2_sec-t1_sec
    print 'Done!'
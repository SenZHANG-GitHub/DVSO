
------
## Some notes by Sen Zhang (The University of Sydney)

------

## (PointHessian)->residuals and (EFPoint)->residualsAll 
+   In FullSystem/, we use (PointHessian)->residuals
+   In OptimizationBackend/, we use (EFPoint)->residualsAll
+   Two substitutes: activeResiduals and efResidual

-------

## Table of Content
[(PointHessian)->residuals](#residuals)

[(EFPoint)->residualsAll](#residualsAll)

[(FullSystem)->activeResiduals](#activeRes)

[(EFResidual)->efResidual](#efRes)

[Other parts](#others)

-------

<a name="residuals"/>

## Where to use (PointHessian)->residuals

**FullSystem::activatePointsMT()**
+   When activating a point (PointHessian* newpoint() in optimized
```cpp
    for (PointFrameResidual* r: newpoint->residuals)
        ef->insertResidual(r);
```

**FullSystem::flagPointsForRemoval()**
+   When determining whether a point should be removed：Descard points behind the camera or without residualss
```cpp
    if (ph->idepth_scaled < 0 || ph->residuals.size() == 0)
```
+   When ph->isOOB() and ph->isInlierNew(), linearize all residuals of ph for later marginalization
```cpp
    for (PointFrameResidual* r : ph->residuals)
    {
        r->resetOOB()
        r->linearize() or r->linearizeVirtualStereo
        r->applyRes(true) 
        ...
    }
```

**FullSystem::makeKeyFrame()**
+   When constructing the residuals between the current frame fh and the active points in previous keyframes
```cpp
    PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
    r->setState(ResState::IN);
    ph->residuals.push_back(r);
    e->insertResidual(r);
    ...
```

**FullSystem::initializeFromInitializer()**
+   When adding the virtual stereo residual of the activated points in the first frame during initialization
```cpp
    firstFrame->pointHessian.push_back(ph);
    ef->insertPoint(ph);
    PointFrameResidual* r = new PointFrameResidual(ph, ph->host, ph->host->frame_right);
    r->state_NewEnery = r->state_enerygy = 0;
    r->state_NewState = ResState::OUTLIER;
    r->setState(ResState::IN);
    r->virtualStereoResFlag = true;
    ph->residuals.push_back(r);
    ef->inserResidual(r);
```

**FullSystem::debugPlotTracking()**
+   When visualizing the residuals of ACTIVE and MARGINALIZED points during debugging
```cpp
    if (ph->status == PointHessan::ACTIVE || ph->status == PointHessian::MARGINALIZED)
    {
        for (PointFrameResidual* r : ph->residuals)
            r->debugPlot()
        ...
    }
```

**FullSystem::marginalizedFrame()**
+   When deleting the residuals of other frames on the frame to be marginalized
```cpp
    for (int = 0; i < >ph->residuals.size(); i++)
    {
        PointFrameResiduals* r = ph->residuals[i];
        if (r->target == frame)
        {
            ...
            ef->dropResidual(r->efResidual);
            deleteOut<PointFrameResidual>(ph->residuals, i);
        }
    }
```

**FullSystem::linearizeAll()**
+   When deleting residuals in toRemove
```cpp
    for (PointFrameResidual* r : toRemove[i])
    {
        PointHessian* ph = r->point;
        ...
        for (int k = 0; k < ph->residuals.size(); k++)
        {
            if (ph->residuals[k] == r)
            {
                ef->dropResidual(r->efResidual);
                deleteOut<PointFrameResidual>(ph->residuals, k);
                nResRemoved++;
                break;
            }
        }
    }
```

**FullSystem::optimize()**
+   When adding the residuals that have not been linearized and marginalized into activeResiduals
```cpp
    for (PointFrameResidual* r : ph->residuals)
    {
        if (!r->efResidual->isLinearized)
        {
            activeResiduals.push_back(r);
            r->resetOOB();
        }
        else
            numLRes++;
    }
```

**FullSystem::removeOutliers()**
+   When marking ph without residuals as PS_DROP (Points to be dropped), and delete from fh->pointHessians
```cpp
    if (ph->residuals.size() == 0)
    {
        fh->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        fh->pointHessians[i] = fh->pointHessians.back();
        fh->pointHessians.pop_back();
        i--;
        numPointsDropped++;
    }
```

**FullSystem::optimizeImmaturePoint()**
+   When successfully activating an immature point p，and add the virtual stereo residual
```cpp
    PointFrameResidual* r = new PointFrameResidual(p, p->host, p->host->frame_right);
    r->state_NewEnergy = r->state_energy = 0;
    r->staet_NewState = ResState::OUTLIER;
    r->setState(ResState::IN);
    r->virtualStereoResFlag = true;
    p->residuals.push_back(r);
```
+   When successfully activating an immature point p，and add the residual 
```cpp
    for (int i=0; i < nres; i++)
        if (residuals[i].state_state == ResState::IN)
        {
            PointFrameResidual* r = new PointFrameResiual(p, p->host, residuals[i].target);
            r->state_NewEnergy = r->state_energy = 0;
            r->state_NewState = ResState::OUTLIER;
            r->setState(ResState::IN);
            p->residuals.push_back(r);
            ...
        }
```

**FullSystem::optimizeImmaturePoint()**
+   ImmaturePointTemporaryResidual* residuals，used for GN optimization in activation
```cpp
    int nres = 0;
    if (fh != point->host)
    {
        residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
        residuals[nres].state_NewState = ResState::OUTLIER;
        residuals[nres].state_state = ResState::IN;
        residuals[nres].target = fh;
        nres++;
    }
    ...
    for (int i=0; i < nres; i++)
    {
        lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i, lastHdd, lastbd, currentIdepth);
        residuals[i].state_state = residuals[i].state_NewState;
        residuals[i].state_energy = residuals[i].state_NewEnergy;
    }
    ...
    for (int iteration=0; iteration < setting_GNItsOnPointActivation>; iteration++)
    {
        ...
        for (int i=0; i < nres; i++)
            newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i, newHdd, newbd, newIdepth);
        ...
        if (newEnergy < lastEnergy)
        {
            ...
            for (int i=0; i < nres; i++)
            {
                residuals[i].state_state = residuals[i].state_NewState;
                residuals[i].state_energy = residuals[i].state_NewEnergy;
            }
            ...
        }
    }
    ...
    for (int i=0; i < nres; i++)
        if (residuals[i].state_state == ResState::IN) 
            numGoodRes++;
```

**PointHessian::release()**
+   When releasing the memory of residuals
```cpp
    for (int i = 0; i < residuals.size(); i++)
        delete residuals[i];
    residuals.clear();
```

**PointHessian::isOOB()**
+   When counting visInToMarg and determining whether it is OOB
```cpp
    for (PointFrameResidual* r : residuals)
    {
        if (r->state_state != ResState::IN)
            continue
        for (FrameHessian*k : toMarg)
            if (r->target == k) 
                visInToMarg++;
    }
    if (residuals.size() >= setting_minGoodActiveResForMarg &&
        numGoodResiduals > setting_minGoodResForMarg + 10 &&
        residuals.size() - visInToMarg < setting_minGoodActiveResForMarg)
    {
        return true;
    }
```

**PointHessian::isInlierNew()**
+   When determining an inlier
```cpp
    return residuals.size() >= setting_minGoodActiveResForMarg &&
           numGoodResiduals >= setting_minGoodResForMarg;
```

-------

<a name="residualsAll"/>

## Where to use (EFPoint)->residualsAll
**AccumulatedSCHessianSSE::addPoint()**
+   When counting ngoodres
```cpp
    for (EFResidual* r : p->residualsAll)
        if (r->isActive())
            ngoodres++;
```
+   When updating accD, accE, accEB
```cpp
    int nFrames2 = nframes[tid] * nframes[tid];
    for (EFResidual* r1 : p->residualsAll)
    {
        if (!r1->isActive())
            continue;
        int r1ht = r1->hostIDX + r1->targetIDX * nframes[tid];
        for (EFResidual* r2 : p->residualsAll)
        {
            if (!r2->isActive())
                continue;
            accD[tid][r1ht + r2->targetIDX * nframes2].update(r1->JpJdF, r2->JpJdF, p->HdiF);
        }
        accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);
        accEB[tid][r1ht].update(r1->JpJdF, p->HdiF * p->bdSumF);
    }
```

**AccumulatedTopHessianSSE::addPoint()**
+   When calculating each residual, and add into Hdd_acc, bd_acc, Hcd_acc
```cpp
    for (EFResidual* r : p->residualsAll)
    {
        ...
        acc[tid][htIDX].update(..);
        acc[tid][htIDX].updateBotRight(..);
        acc[tid][htIDX].updateTopRight(..);
        bd_acc += ..;
        Hdd_acc += ..;
        Hcd_acc += ..;
        nres[tid]++;
    }
```

**EnergyFunctional::~EnergyFunctional()**
+   When destructing variables in ef
```cpp
    for (EFFrame* f : frames)
    {
        for (EFPoint* p : f->points)
        {
            for (EFResidual* r : p->residualsAll)
            {
                r->data->efResidual = 0;
                delete r;
            }
            p->data->efResidual = 0;
            delete p;
        }
        f->data->efFrame = 0;
        delete f;
    }
```

**EnergyFunctional::resubstituteFPt()**
+   When updating the increment of idepth
```cpp
    for (int k=min; k < max; k++)
    {
        EFPoint* p = allPoints[k];
        for (EFResidual* r : p->residualsAll)
            if (r->isActive())
                ngoodres++;
        if (ngoodres == 0)
        {
            p->data->step = 0;
            continue;
        }
        float b = p->bdSumF;
        b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);
        for (EFResidual* r : p->residualsAll)
        {
            if (!r->isActive())
                continue;
            b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
        }
        p->data->step = -b * p->HdiF;
    }
```

**EnergyFunctional::calcEnergyPt()**
+   When calculating the energy of idepth
```cpp
    for (int i=min; i < max ; i++)
    {
        EFPoint* p = allPoints[i];
        float dd = p->deltaF;
        for (EFResidual* r : p->residualsAll)
        {
            ....
            for (int i=((patternNum>>2)<<2; i < patternNum; i++))
            {
                ...
                E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2*r->res_toZeroF[i])));
            }
        }
        E.updateSingle(p->deltaF * p->deltaF * p->priorF);
    }
    E.finish();
    (*stats)[0] += E.A;
```

**EnergyFunctional::insertResidual()**
+   When transforming an input PointFrameResidual* r to EFResidual and save into efPoint
```cpp
    EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
    efr->idxInAll = r->point->efPoint->residualsAll.size();
    r->point->efPoint->residualsAll.push_back(efr);
    ...
    nResiduals++;
    r->efResidual = efr;
```

**EnergyFunctional::dropResidual()**
+   When deleting a EFREsidual from EFPoint
```cpp
    EFPoint* p = r->point;
    assert(r == p->residualsAll[r->idxInAll]);

    p->residualsAll[r->idxInAll] = p->residualsAll.back();
    p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
    p->residualsAll.pop_back();
    ...
    nResiduals--;
    r->data->efResidual=0;
    delete r;
```

**EnergyFunctional::marginalizePointsF()**
+  When updating the connectivityMap of points in PS_MARGINALIZE
```cpp
    if (p->stateFlag == EFPointStatus::PS_MARGINALIZE)
    {
        ...
        for (EFResidual* r : p->residualsAll)
        {
            if (r->isActive() && r->data->virtualStereoResFlag == false)
                connectivityMap[((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
        }
        allPointsToMarg.push_back(p);
    }
```

**EnergyFunctional::removePoint()**
+   When deleting the residuals of the points to be removed
```cpp
    for (EFResidual* r : p->residualsAll)
        dropResidual(r);
```

**EnergyFunctional::makeIDX()**
+   When updating idx for all points 
```cpp
    allPoints.clear();
    for (EFFrame* f : frames)
    {
        for (EFPoint* p : f->points)
        {
            allPoints.push_back(p);
            for (EFResidual* r : p->residualsAll)
            {
                r->hostIDX = r->host->idx;
                r->targetIDX = r->target->idx;

                if (r->data->virtualStereoResFlag == true)
                    r->targetIDX = frames[frames.size()-1]->idx;
            }
        }
    }
```


-------

<a name="activeRes"/>

## Where to use activeResiduals
**FullSystem.h**
+   When defining activeResiduals
```cpp
    std::vector<PointFrameResidual*> activeResiduals;
```

**FullSystem::linearizeAll_Reductor()**
+   When linearlizing each r in activeResiduals
```cpp
    for (int k=min; k < max; k++)
    {
        PointFrameResidual* r = activeResiduals[k];
        if (r->virtualStereoResFlag == true)
            (*stats)[0] += r->linearizeVirtualStereo(&Hcalib);
        else
            (*stats)[0] += r->linearize(&Hcalib);
        
        if (fixLinearization)
        {
            r->applyRes(true)
            if (r->efResidual->isActive())
                ...
            else
                toRemove[tid].push_back(activeResiduals[k]);
        }
    }
```

**FullSystem::applyRes_Reductor()**
+   When applying applyRes() to each r in activeResiduals
```cpp
    for (int k=min; k < max; k++)
        activeResiduals[k]->applyRes(true);
```

**FullSystem::setNewFrameEnergyTH()**
+   When accumulating allResVec for newFrame->frameEnergyTH
```cpp
    allResVec.clear();
    allResVec.reserve(activeResiduals.size() * 2);
    FrameHessian* newFrame = frameHessians.back();
    for (PointFrameResidual* r : activeResiduals)
    {
        if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame)
            allResVec.push_back(r->state_NewEnergyWithOutlier);
    }
    ...
```

**FullSystem::linearizeAll()**
＋  As the arguments of FullSystem::linearizeAll_Reductor(), see above
＋  When fixLinearization is true, update the lastResiduals of the corresponding ph of activeResiduals->r
```cpp
    if (fixLinearization)
    {
        for (PointFrameResidual* r : activeResiduals)
        {
            PointHessian* ph = r->point;
            if (ph->lastResiduals[0].first == r)
                ph->lastResiduals[0].second = r->state_state;
            else if (ph->lastResiduals[1].first == r)
                ph->lastResiduals[1].second = r->state_state;
        }
        ...
    }
```
＋  As the arguments of FullSystem::applyRes_Reductor(), see above.

**FullSystem::optimize()**
+   When setting all unlinearized residuals to activeResiduals
```cpp
    activeResiduals.clear();
    int numPoints = 0;
    int numLRes = 0;
    for (FrameHessian* fh : frameHessians)
    {
        for (PointHessian* ph : fh->pointHessians)
        {
            for (PointFrameResidual* r : ph->residuals)
            {
                if (!r->efResidual->isLinearized)
                {
                    activeResiduals.push_back(r);
                    r->resetOOB();
                }
                else 
                numLRes++;
            }
            numPoints++;
        }
    }
```


-------

<a name="efRes"/>

## Where to use efResidual
**CoarseTracker::makeCoarseDepthL0()**
+  When determining whether a point should be projected to the current frame to get the idepth estimate
```cpp
    PointFrameResidual* r = ph->lastResiduals[0].first;
    assert(r->efResidual->isActive() && r->target == lastRef);
    int u = r->centerProjectedTo[0] + 0.5f;
    int v = r->centerProjectedTo[1] + 0.5f;
    float new_idepth = r->centerProjectedTo[2];
    ...
```

**FullSystem::flagPointsForRemoval()**
+  When marking an inlier point to be marginalized, we want to linearize its residue at current step
```cpp
    if (ph->isInlierNew())
    {
        ...
        for (PointFrameResidual* r : ph->residuals)
        {
            r->resetOOB();
            if (r->virtualStereoResFlag == true)
                r->linearizeVirtualStereo(&Hcalib);
            else
                r->linearize(&Hcalib);
            
            r->efResidual->isLinearized = false;
            r->applyRes(true);
            if (r->efResidual->isActive())
            {
                r->efResidual->fixLinerizationF(ef);
                ngoodRes++;
            }
        }
        ...
    }
```

**FullSystem::marginalizeFrame()**
+   When deleting the residuals between the frame to be marginalized and other frames
```cpp
    for (int i=0; i < ph->residuals.size(); i++)
    {
        PointFrameResidual* r = ph->residuals[i];
        if (r->target == frame)
        {
            ...
            ef->dropResidual(r->efResidual);
            deleteOut<PointFrameResidual>(ph->residuals, i);
            break;
        }
    }
```

**FullSystem::linearizeAll_Reductor()**
+   When fixLinearization is true, determine whether current residual should be discarded
```cpp
    if (fixLinearization)
    {
        r->applyRes(true);
        if (r->efResidual->isActive())
        {
            ... // targetPrecalc, p->maxRelBaseline, p->numGoodResiduals
        }
        else 
            toRemove[tid].push_back(activeResiduals[k])
    }
```


**FullSystem::linearizeAll()**
+   When fixLinearization is true, delete the resisuals in toRemove
```cpp
    if (fixLinearization)
    {
        ...
        for (PointFrameResidual* r : toRemove[i])
        {
            PointHessian* ph = r->point;
            ...
            for (int k=0; k < ph->residuals.size(); k++)
            {
                if (ph->residuals[k] == r)
                {
                    ef->dropResidual(r->efResidual);
                    deleteOut<PointFrameResidual>(ph->residuals, k);
                    nResRemoved++;
                    break;
                }
            }
        }
    }
```

**FullSystem::optimize()**
+   When setting all unlinearized residuals to activeResiduals
```cpp
    activeResiduals.clear();
    int numPoints = 0;
    int numLRes = 0;
    for (FrameHessian* fh : frameHessians)
    {
        for (PointHessian* ph : fh->pointHessians)
        {
            for (PointFrameResidual* r : ph->residuals)
            {
                if (!r->efResidual->isLinearized)
                {
                    activeResiduals.push_back(r);
                    r->resetOOB();
                }
                else 
                numLRes++;
            }
            numPoints++;
        }
    }
```

**PointFrameResidual::~PointFrameResidual()**
+   When deleting r，its efResidual should already be set to ０
```cpp
    assert(efResidual==0);
```

**PointFrameResidual::PointFrameResidual()**
＋  During initializatoin，initliaze efResidual to ０ as well

**PointFrameResidual::applyRes()**
+   When copyJacobians is true，update isActiveAndIsGoodNEW
```cpp
    if (state_state == ResState::OOB)
    {
        assert(!efResidual->isActiveAndIsGoodNEW);
        return;
    }
    if (state_NewState == ResState::IN)
    {
        efResidual->isActiveAndIsGoodNEW = true;
        efResidual->takeDataF();
    }
    else
        efResidual->isActiveAndIsGoodNEW = false;
```

**AccumulatedSCHessianSSE::addPoint()**
+   When counting ngoodres
```cpp
    for (EFResidual* r : p->residualsAll)
    {
        if (r->isActive())
            ngoodres++;
    }
```
+   When updating accD, accE, accEB
```cpp
    for(EFResidual* r1 : p->residualsAll)
	{
		if(!r1->isActive()) continue;
		int r1ht = r1->hostIDX + r1->targetIDX*nframes[tid];

		for(EFResidual* r2 : p->residualsAll)
		{
			if(!r2->isActive()) continue;

			accD[tid][r1ht+r2->targetIDX*nFrames2].update(r1->JpJdF, r2->JpJdF, p->HdiF);
		}

		accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);
		accEB[tid][r1ht].update(r1->JpJdF,p->HdiF*p->bdSumF);
	}
```

**AccumulatedTopHessianSSE::addPoint()**
+   Accumulate bd_acc, Hdd_acc, Hcd_acc for each r in p->residualsAll
```cpp
    for(EFResidual* r : p->residualsAll)
    {
        ... // update bd_acc, Hdd_acc, Hcd_acc
    }
```

**EnergyFunctional.cpp**
+   Highly related to residualsAll(). See the calls of residualsAll()
```cpp
    for (EFResidual* r : p->residualsAll)
```

**EFResidual::takeDataF()**
+   Update JpJdF

**EFRedisual::fixLinearizationF()**
+   Set isLinearized as true


----

<a name="others"/>

## Other parts

**efResidual->isLinearized**
+    Default: false
+    Set to true in efResidual::fixLinearizationF()

**efResidual::fixLinearizationF()**
+    Be careful with  the adHTdeltaF indices
```cpp
    Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];
```
+ IN FullSystem::flagPointsForRemoval()
    +    After we linearize the residual to be marg
```cpp
    if (r->efResidual->isActive())
    {
    	r->efResidual->fixLineaization(ef);
	    ngoodRes++;
    }
```
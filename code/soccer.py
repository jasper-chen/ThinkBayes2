
"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot


class Soccer(thinkbayes2.Suite):
    """Represents hypotheses about."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: hypothetical goal-scoring rate in goals per game
        data: time between goals in minutes
        """

        x = data
        lam = hypo/90
        like = thinkbayes2.EvalExponentialPdf(x, lam)
        return like

    def PredRemaining(self, rem_time, score):
        """Plots the predictive distribution for final number of goals.

        rem_time: remaining time in the game in minutes
        score: number of goals already scored
        """
        # TODO: fill this in
        metapmf = thinkbayes2.Pmf()
        for lam, prob in self.Items():
            print (lam,prob)
            lt = lam * rem_time / 90
            pmf = thinkbayes2.MakePoissonPmf(lt, 12)
            thinkplot.Pdf(pmf, linewidth=1, alpha=0.2, color='purple')
            metapmf[pmf] = prob

        mix = thinkbayes2.MakeMixture(metapmf)
        mix += 2
        thinkplot.Hist(mix)
        thinkplot.Show()


def main():
    #start with a uniform distribution for lambda in goals per game
    hypos = numpy.linspace(0, 12, 201)
    suite = Soccer(hypos)

    thinkplot.PrePlot(4)
    thinkplot.Pdf(suite, label='prior')
    print('prior mean', suite.Mean())

    #construct a prior using a pseudo-observation

    suite.Update(69)
    thinkplot.Pdf(suite, label='prior 2')
    print('after one goal', suite.Mean())

    #update with real data

    suite.Update(11)
    thinkplot.Pdf(suite, label='posterior 1')
    print('after one goal', suite.Mean())

    suite.Update(12)
    thinkplot.Pdf(suite, label='posterior 2')
    print('after two goals', suite.Mean())

    thinkplot.Show(xlabel='$\lambda$ in goals per game')

    suite.PredRemaining(67,2)


if __name__ == '__main__':
    main()

function [err, delta] = computeError(out, y)  err = (1/2)*((y-out)'*(y-out));  delta = y-out;end
#!/bin/sh
cd /data/beegfs/home/gosalcds/master_den
git add --all
timestamp() 
{
  date +"at %H:%M:%S on %d/%m/%Y"
}
git commit -am "Regular auto-commit $(timestamp)"
git push

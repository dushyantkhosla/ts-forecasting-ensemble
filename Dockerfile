FROM centos:latest
MAINTAINER Dushyant Khosla <dushyant.khosla@yahoo.com

COPY environment.yml environment.yml
COPY start.sh /etc/profile.d/

# Install
RUN yum -y install tmux \
                   bzip2 \
                   wget \
                   which \
                   curl

# Get and Install Conda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh \
  && bash miniconda.sh  -b -p /miniconda \
  && rm miniconda.sh

ENV PATH="/miniconda/bin:${PATH}"
RUN conda config --add channels conda-forge
RUN conda env create -f environment.yml --quiet


# Install latest version of Git
RUN yum -y remove git \
  && wget https://github.com/git/git/archive/v2.15.1.tar.gz -O git.tar.gz \
  && tar -zxf git.tar.gz \
  && rm -f git.tar.gz

WORKDIR git-2.15.1

RUN yum -y groupinstall "Development Tools"
RUN yum -y install zlib-devel \
                   perl-devel \
                   perl-CPAN \
                   curl-devel

RUN make configure \
  && ./configure --prefix=/usr/local \
  && make install \
  && rm -rf /git-2.15.1/

# Get Fish and OMF
WORKDIR /etc/yum.repos.d/
RUN wget https://download.opensuse.org/repositories/shells:fish:release:2/CentOS_7/shells:fish:release:2.repo \
  && yum install -y fish

# Clean up
RUN yum -y autoremove \
  && yum clean all \
  && rm -rf /var/cache/yum


# Copy reference material
RUN mkdir /home/ts-reference

COPY /data /home/ts-reference/
COPY /notebooks /home/ts-reference/

# Start Here
WORKDIR /home/

EXPOSE 8080
CMD /usr/bin/bash

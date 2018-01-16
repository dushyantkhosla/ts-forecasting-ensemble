FROM centos:latest
MAINTAINER Dushyant Khosla <dushyant.khosla@yahoo.com

# === COPY FILES ===

COPY environment.yml /root/environment.yml
COPY start.sh /etc/profile.d/

# === SET ENVIRONMENT VARIABLES ===

ENV PATH="/miniconda/bin:${PATH}"
ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8

# === INSTALL DEPENDENCIES ===

WORKDIR /root
RUN yum -y install bzip2 \
                   curl \
                   curl-devel \
                   perl-devel \
                   perl-CPAN \
                   tmux \
                   wget \
                   which \
                   zlib-devel \
	&& yum -y groupinstall "Development Tools" \
&& yum -y remove git \
	&& wget https://github.com/git/git/archive/v2.15.1.tar.gz -O git.tar.gz \
	&& tar -zxf git.tar.gz \
	&& rm -f git.tar.gz \
&& wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh \
	&& bash miniconda.sh  -b -p /miniconda \
	&& conda config --append channels conda-forge \
	&& conda env create -f environment.yml \
	&& conda clean -i -l -t -y \
	&& rm miniconda.sh \
&& wget https://download.opensuse.org/repositories/shells:fish:release:2/CentOS_7/shells:fish:release:2.repo -P /etc/yum.repos.d/ \
	&& yum install -y fish \
&& yum -y autoremove \
  	&& yum clean all \
	&& rm -rf /var/cache/yum

WORKDIR /root/git-2.15.1
RUN make configure \
	&& ./configure --prefix=/usr/local \
	&& make install \
	&& rm -rf /git-2.15.1

# === INITIALIZE ===

# Copy reference material
COPY /data /home/data/
COPY /notebooks /home/notebooks/

WORKDIR /home/
EXPOSE 8080
CMD /usr/bin/bash
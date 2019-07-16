elifePipeline {
    node('containers-jenkins-plugin') {
        def commit

        stage 'Checkout', {
            checkout scm
            commit = elifeGitRevision()
            baseGrobidTag = sh(
                script: 'bash -c "source .env && echo \\$BASE_GROBID_TAG"',
                returnStdout: true
            ).trim()
            echo "baseGrobidTag: ${baseGrobidTag}"
            assert baseGrobidTag != ''
        }

        stage 'Build and run tests', {
            try {
                sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-build-and-test"
            } finally {
                sh "make ci-clean"
            }
        }

        elifeMainlineOnly {
            stage 'Merge to master', {
                elifeGitMoveToBranch commit, 'master'
            }

            stage 'Push unstable sciencebeam-trainer-delft image', {
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft', commit)
                def unstable_image = image.addSuffixAndTag('_unstable', commit)
                unstable_image.tag('latest').push()
                unstable_image.push()
            }

            stage 'Push unstable sciencebeam-grobid-delft image', {
                def image = DockerImage.elifesciences(this, 'sciencebeam-grobid-delft', commit)
                def unstable_image = image.addSuffixAndTag('_unstable', commit)
                unstable_image.tag('latest').push()
                unstable_image.push()
            }
        }
    }
}
